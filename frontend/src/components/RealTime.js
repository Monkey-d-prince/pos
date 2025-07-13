import { useEffect, useRef, useState } from "react";
import "./RealTime.css";

//const WS_URL = "ws://localhost:8000/ws/realtime";
const WS_URL = "wss://40.82.136.174:8000/ws/realtime";

function RealTime() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ws = useRef(null);
  const reconnectTimeout = useRef(null);
  const [feedback, setFeedback] = useState("");
  const [connected, setConnected] = useState(false);
  const [postureType, setPostureType] = useState("");
  const [postureQuality, setPostureQuality] = useState("");
  const [videoDims, setVideoDims] = useState({ width: 480, height: 360 });
  const busy = useRef(false);

  // Robust WebSocket connect/reconnect
  const connectWebSocket = () => {
    // Only close if open or connecting
    if (ws.current && (ws.current.readyState === 0 || ws.current.readyState === 1)) {
      ws.current.onclose = null; // prevent triggering reconnect twice
      ws.current.close();
    }
    ws.current = new window.WebSocket(WS_URL);
    ws.current.onopen = () => setConnected(true);
    ws.current.onclose = () => {
      setConnected(false);
      // Prevent multiple reconnects
      if (!reconnectTimeout.current) {
        reconnectTimeout.current = setTimeout(() => {
          reconnectTimeout.current = null;
          connectWebSocket();
        }, 1000);
      }
    };
    ws.current.onerror = () => {
      ws.current.close();
    };
    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.posture_type) setPostureType(data.posture_type);
      if (data.posture_quality) setPostureQuality(data.posture_quality);
      if (data.issues && data.issues.length > 0) {
        setFeedback(data.issues.join(" | "));
      } else if (data.good_posture) {
        setFeedback("Good posture detected!");
      } else {
        setFeedback("");
      }
      // Draw only shoulder, neck (nose), and back (hip) points with enhanced visuals
      if (data.landmarks && canvasRef.current && videoRef.current) {
        const ctx = canvasRef.current.getContext("2d");
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
        ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
        // Indices for nose, left/right shoulder, left/right hip
        const indices = [0, 11, 12, 23, 24];
        const colors = ["#007bff", "#ff4d4f", "#ff4d4f", "#00c9a7", "#00c9a7"];
        indices.forEach((idx, i) => {
          const lm = data.landmarks[idx];
          if (lm) {
            ctx.beginPath();
            ctx.arc(lm.x * canvasRef.current.width, lm.y * canvasRef.current.height, 10, 0, 2 * Math.PI);
            ctx.shadowColor = colors[i];
            ctx.shadowBlur = 10;
            ctx.fillStyle = colors[i];
            ctx.fill();
            ctx.shadowBlur = 0;
          }
        });
        // Draw lines: neck to shoulders, shoulders to hips
        const getXY = (idx) => [data.landmarks[idx].x * canvasRef.current.width, data.landmarks[idx].y * canvasRef.current.height];
        // Neck (nose) to left/right shoulder
        ctx.strokeStyle = "#ffa502";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(...getXY(0));
        ctx.lineTo(...getXY(11));
        ctx.moveTo(...getXY(0));
        ctx.lineTo(...getXY(12));
        ctx.stroke();
        // Shoulders to hips
        ctx.strokeStyle = "#00c9a7";
        ctx.lineWidth = 5;
        ctx.beginPath();
        ctx.moveTo(...getXY(11));
        ctx.lineTo(...getXY(23));
        ctx.moveTo(...getXY(12));
        ctx.lineTo(...getXY(24));
        ctx.stroke();
        // Angle marks with background
        ctx.font = "bold 20px 'Segoe UI', Arial";
        ctx.fillStyle = "#fff";
        ctx.globalAlpha = 0.7;
        ctx.fillRect(8, 18, 180, 140);
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = "#007bff";
        if (data.back_angle !== undefined) ctx.fillText(`Back: ${data.back_angle.toFixed(1)}°`, 20, 40);
        if (data.neck_angle !== undefined) ctx.fillText(`Neck: ${data.neck_angle.toFixed(1)}°`, 20, 70);
        if (data.knee_angle !== undefined) ctx.fillText(`Knee: ${data.knee_angle.toFixed(1)}°`, 20, 100);
        if (data.shoulder_diff !== undefined) ctx.fillText(`Shoulder diff: ${data.shoulder_diff.toFixed(2)}`, 20, 130);
        if (data.hip_diff !== undefined) ctx.fillText(`Hip diff: ${data.hip_diff.toFixed(2)}`, 20, 160);
      }
    };
  };

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (ws.current) ws.current.close();
      if (reconnectTimeout.current) clearTimeout(reconnectTimeout.current);
    };
  }, []);

  useEffect(() => {
    let interval;
    const sendFrame = () => {
      if (
        videoRef.current &&
        ws.current &&
        ws.current.readyState === 1 &&
        !busy.current
      ) {
        busy.current = true;
        const canvas = document.createElement("canvas");
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL("image/jpeg");
        ws.current.send(
          JSON.stringify({ image: dataUrl, posture_type: "auto" })
        );
      }
    };
    if (connected) {
      interval = setInterval(sendFrame, 1); // Send every 500ms for less lag
    }
    return () => clearInterval(interval);
  }, [connected]);

  // Mark not busy when a message is received (fix: always call busy.current = false, even if error)
  useEffect(() => {
    if (!ws.current) return;
    const prevOnMessage = ws.current.onmessage;
    ws.current.onmessage = (event) => {
      busy.current = false;
      try {
        if (prevOnMessage) prevOnMessage(event);
      } catch (e) {
        // ignore
      }
    };
    ws.current.onerror = (err) => {
      busy.current = false;
      ws.current.close();
    };
    // No cleanup needed, ws.current will be replaced on reconnect
  }, [connected]);

  const startWebcam = async () => {
    if (navigator.mediaDevices.getUserMedia) {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      videoRef.current.srcObject = stream;
      // Wait for video to be ready
      videoRef.current.onloadedmetadata = () => {
        setVideoDims({
          width: videoRef.current.videoWidth,
          height: videoRef.current.videoHeight,
        });
      };
    }
  };

  useEffect(() => {
    startWebcam();
  }, []);

  return (
    <div className="realtime-container">
      <div className="realtime-video-box" style={{ width: videoDims.width, height: videoDims.height }}>
        <video
          ref={videoRef}
          width={videoDims.width}
          height={videoDims.height}
          autoPlay
          className="realtime-video"
        />
        <canvas
          ref={canvasRef}
          width={videoDims.width}
          height={videoDims.height}
          className="realtime-canvas"
        />
      </div>
      <div className={`realtime-posture-quality ${postureQuality.includes("Good") ? "good" : postureQuality.includes("Okay") ? "okay" : "bad"}`}>
        {postureQuality}
      </div>
      <div className={`realtime-feedback ${feedback.includes("Good") ? "good" : feedback ? "bad" : ""}`}>
        {feedback}
      </div>
      <div className="realtime-type">
        Posture type: <b>{postureType}</b>
      </div>
      <div className="realtime-status">
        <span className={connected ? "good" : "bad"}>
          {connected ? "WebSocket Connected" : "WebSocket Disconnected"}
        </span>
      </div>
    </div>
  );
}

export default RealTime;
