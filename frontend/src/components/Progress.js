import ProgressBar from "progressbar.js"
import {useEffect} from "react"
import "./Progress.css"

function Progress() {

    useEffect(()=>{

        var circle = new ProgressBar.Line('#progress-bar', {
            color: '#ffffff',
            duration: 5000,
            easing: 'easeInOut'
        });
    
        circle.animate(1);
    })
    return (
    <div className="progress">
        <h1>당신의 취향에 맞는 와인 찾는 중 !</h1>
        <div className="progress-container">
        <div id="progress-bar">

        </div>
        </div>
        <div>icon</div>
    </div>
    );
  }
  
  export default Progress;
  