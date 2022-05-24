import { useRef, useState } from "react";
import axios from "axios";
import Title from "../components/Title";
import "./Survey.scss"
import ProgressBar from "progressbar.js";
// import ProgressBar from "../components/ProgressBar";

function Survey() {
    const [checked, setChecked] = useState({})
    const checkedItemHandler = ({target})=>{
            checked[target.name] = target.value
            setChecked(checked)
            console.log(checked)
    }
    const onSubmit = (e)=>{
        e.preventDefault()
        // console.log(123)
        // let bar = new ProgressBar.Line('#progress-bar', {easing: 'easeInOut'});
        // bar.animate(1);
        // let data = axios.post("http://127.0.0.1:8000/api/get-recommend",{})
    }

    return (
      <div id="form-wrapper">
          <Title/>
          <form action="http://127.0.0.1:8000/api/get-recommend"  method="POST">
		<div id="debt-amount-slider">
            <div>
			<input type="radio" name="type-debt-amount" id="type-1" value="1" onChange={checkedItemHandler} required/>
			<label htmlFor="type-1" data-debt-amount="Red"></label>
			<input type="radio" name="type-debt-amount" id="type-2" value="2" onChange={checkedItemHandler} required/>
			<label htmlFor="type-2" data-debt-amount="White"></label>
			<input type="radio" name="type-debt-amount" id="type-3" value="3" onChange={checkedItemHandler} required/>
			<label htmlFor="type-3" data-debt-amount="Sparkling"></label>
			<input type="radio" name="type-debt-amount" id="type-4" value="4" onChange={checkedItemHandler} required/>
			<label htmlFor="type-4" data-debt-amount="Rose"></label>
			<input type="radio" name="type-debt-amount" id="type-5" value="5" onChange={checkedItemHandler} required/>
			<label htmlFor="type-5" data-debt-amount="ETC"></label>
			<div id="debt-amount-pos"></div>
            </div>
            <div>
			<input type="radio" name="sweet-debt-amount" id="sweet-1" value="1" onChange={checkedItemHandler} required/>
			<label for="sweet-1" data-debt-amount="None"></label>
			<input type="radio" name="sweet-debt-amount" id="sweet-2" value="2" onChange={checkedItemHandler} required/>
			<label for="sweet-2" data-debt-amount=""></label>
			<input type="radio" name="sweet-debt-amount" id="sweet-3" value="3" onChange={checkedItemHandler} required/>
			<label for="sweet-3" data-debt-amount="Moderately"></label>
			<input type="radio" name="sweet-debt-amount" id="sweet-4" value="4" onChange={checkedItemHandler} required/>
			<label for="sweet-4" data-debt-amount=""></label>
			<input type="radio" name="sweet-debt-amount" id="sweet-5" value="5" onChange={checkedItemHandler} required/>
			<label for="sweet-5" data-debt-amount="Much"></label>
			<div id="debt-amount-pos"></div>
            </div>
            <div>
			<input type="radio" name="acid-debt-amount" id="acid-1" value="1" onChange={checkedItemHandler} required/>
			<label for="acid-1" data-debt-amount="None"></label>
			<input type="radio" name="acid-debt-amount" id="acid-2" value="2" onChange={checkedItemHandler} required/>
			<label for="acid-2" data-debt-amount=""></label>
			<input type="radio" name="acid-debt-amount" id="acid-3" value="3" onChange={checkedItemHandler} required/>
			<label for="acid-3" data-debt-amount="Moderately"></label>
			<input type="radio" name="acid-debt-amount" id="acid-4" value="4" onChange={checkedItemHandler} required/>
			<label for="acid-4" data-debt-amount=""></label>
			<input type="radio" name="acid-debt-amount" id="acid-5" value="5" onChange={checkedItemHandler} required/>
			<label for="acid-5" data-debt-amount="Much"></label>
			<div id="debt-amount-pos"></div>
            </div>
            <div>
			<input type="radio" name="body-debt-amount" id="body-1" value="1" onChange={checkedItemHandler} required/>
			<label for="body-1" data-debt-amount="None"></label>
			<input type="radio" name="body-debt-amount" id="body-2" value="2" onChange={checkedItemHandler} required/>
			<label for="body-2" data-debt-amount=""></label>
			<input type="radio" name="body-debt-amount" id="body-3" value="3" onChange={checkedItemHandler} required/>
			<label for="body-3" data-debt-amount="Moderately"></label>
			<input type="radio" name="body-debt-amount" id="body-4" value="4" onChange={checkedItemHandler} required/>
			<label for="body-4" data-debt-amount=""></label>
			<input type="radio" name="body-debt-amount" id="body-5" value="5" onChange={checkedItemHandler} required/>
			<label for="body-5" data-debt-amount="Much"></label>
			<div id="debt-amount-pos"></div>
            </div>
            <div>
			<input type="radio" name="tannin-debt-amount" id="tannin-1" value="1" onChange={checkedItemHandler} required/>
			<label for="tannin-1" data-debt-amount="None"></label>
			<input type="radio" name="tannin-debt-amount" id="tannin-2" value="2" onChange={checkedItemHandler} required/>
			<label for="tannin-2" data-debt-amount=""></label>
			<input type="radio" name="tannin-debt-amount" id="tannin-3" value="3" onChange={checkedItemHandler} required/>
			<label for="tannin-3" data-debt-amount="Moderately"></label>
			<input type="radio" name="tannin-debt-amount" id="tannin-4" value="4" onChange={checkedItemHandler} required/>
			<label for="tannin-4" data-debt-amount=""></label>
			<input type="radio" name="tannin-debt-amount" id="tannin-5" value="5" onChange={checkedItemHandler} required/>
			<label for="tannin-5" data-debt-amount="Much"></label>
			<div id="debt-amount-pos"></div>
            </div>
		</div>
	<input type="submit" className="submit-info" value="Done" onSubmit={onSubmit}/>
	</form>
    {/* <ProgressBar/> */}
    <div id="progress-bar">

    </div>
      </div>
    );
  }
  
  export default Survey;
