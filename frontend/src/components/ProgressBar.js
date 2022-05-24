// import "./ProgressBar.css"

function ProgressBar() {
    var circle = new ProgressBar.Circle('#progress', {
        color: '#FCB03C',
        duration: 3000,
        easing: 'easeInOut'
    });

    circle.animate(1);
    return (
    <div className="progress-bar">
        {/* <div className="progress-bar__bar active"></div> */}
        
    </div>
    );
  }
  
  export default ProgressBar;
  