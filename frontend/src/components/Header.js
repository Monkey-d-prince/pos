import { NavLink } from "react-router-dom";
import "./Header.css";

const Header = () => (
  <header className="custom-header sticky">
    <h1>Posture Detection App</h1>
    <nav>
      <ul>
        <li><NavLink to="/" className={({isActive}) => isActive ? "nav-link active" : "nav-link"}>Real-Time</NavLink></li>
    </ul>
    </nav>
  </header>
);

export default Header;
