import Title from "../components/Title";
import {
  BrowserRouter,
  Routes,
  Route,
  Link
} from "react-router-dom";

function HomePage() {
  return (
    <div className="home-page">
      <Title/>
      <div className="start-survey">
      <Link to="survey">
          icon
        </Link>
      </div>
      <div className="help-text">
        <span>간단한 5가지 질문에 답해주세요!</span><br/>
        <span>취향에 맞는 와인을 추천해드립니다</span>
      </div>
    </div>
  );
}

export default HomePage;
