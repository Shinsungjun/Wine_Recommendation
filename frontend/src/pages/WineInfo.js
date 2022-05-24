import { useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import Title from "../components/Title";
import "./WineInfo.css"

function WineInfo() {
    const location = useLocation()
    const data = location.state

    return (
      <div className="wine-info">
          <Title/>
          <h3>당신에게 딱 맞는 와인</h3>
          <h1>{data.name}</h1>
          <img src={data.path}></img>
          <div>
              <span >{data.ml}</span>
              <span className="meta">{data.abv}</span>
              <span className="meta">{data.nation}</span>
              <span className="meta">{data.degree}</span>
              <span className="meta">{data.price}</span>
          </div>
          <div className="border-box"></div>
          <div className="info-container">
            <div className="column">
            <h6>sweet</h6>
            <h6>acid</h6>
            <h6>body</h6>
            <h6>tannin</h6>

            </div>
            <div className="value">
                
            <h6>tannin</h6>
            <h6>tannin</h6>
            <h6>tannin</h6>
            <h6>tannin</h6>
            </div>
          </div>
      </div>
    );
  }
  
  export default WineInfo;
