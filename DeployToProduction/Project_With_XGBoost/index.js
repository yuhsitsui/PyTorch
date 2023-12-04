async function runExample() {

    var x = new Float32Array( 1, 15 )

    var x = [];

     x[0] = document.getElementById('box1').value;
     x[1] = document.getElementById('box2').value;
     x[2] = document.getElementById('box3').value;
     x[3] = document.getElementById('box4').value;
     x[4] = document.getElementById('box5').value;
     x[5] = document.getElementById('box6').value;
     x[6] = document.getElementById('box7').value;
     x[7] = document.getElementById('box8').value;
     x[8] = document.getElementById('box9').value;
     x[9] = document.getElementById('box10').value;
     x[10] = document.getElementById('box11').value;
     x[11] = document.getElementById('box12').value;
     x[12] = document.getElementById('box13').value;
     x[13] = document.getElementById('box14').value;
     x[14] = document.getElementById('box15').value;

    let tensorX = new ort.Tensor('float32', x, [1, 15] );
    let feeds = {float_input: tensorX};

    let session = await ort.InferenceSession.create('xgboost_ort.onnx');
    
   let result = await session.run(feeds);
   let outputData = result.variable.data;

  outputData = parseFloat(outputData).toFixed(2)

   let predictions = document.getElementById('predictions');

  predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
   <table>
     <tr>
       <td>  Car MSRP  </td>
       <td id="td0">  ${outputData}  </td>
     </tr>
  </table>`;
    


}

async function runExample1() {

  var x = new Float32Array( 1, 15 )

  var x = [];

   x[0] = 20; // Make
   x[1] = 227; // Model
   x[2] = document.getElementById('years').value; // Year
   x[3] = 18; // Front Wheel Size (in)
   x[4] = document.getElementById('horsepower').value; // SAE Net Horsepower @ RPM
   x[5] = 20; // Displacement
   x[6] = 4; // Engine Type
   x[7] = 72; // Width, Max w/o mirrors (in)
   x[8] = 57; // Height, Overall (in)
   x[9] = 185; // Length, Overall (in)
   x[10] = document.getElementById('mileage').value; // Gas Mileage
   x[11] = 1.5; // Drivetrain
   x[12] = 5; // Passenger Capacity
   x[13] = 4; // Passenger Doors
   x[14] = 2; // Body Style

  let tensorX = new ort.Tensor('float32', x, [1, 15] );
  let feeds = {float_input: tensorX};

  let session = await ort.InferenceSession.create('xgboost_ort.onnx');
  
 let result = await session.run(feeds);
 let outputData = result.variable.data;

outputData = parseFloat(outputData).toFixed(2)

 let predictions = document.getElementById('predictions1');

predictions.innerHTML = ` <hr> Got an output tensor with values: <br/>
 <table>
   <tr>
     <td>  Car MSRP  </td>
     <td id="td0">  ${outputData}  </td>
   </tr>
</table>`;
  


}
