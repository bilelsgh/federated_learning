
function addData(chart, label, data, i) {
    chart.data.labels.push(label);

    chart.data.datasets[i].data.push(data); // accuracy
    
    chart.update();
}

function removeData(chart) {
    chart.data.labels.pop();
    chart.data.datasets.forEach((dataset) => {
        dataset.data.pop();
    });
    chart.update();
}

function getGlobalAcc(_data, chart, entity, best_data, best_model){
    
    // Add data
    let data = _data.slice(chart.data.datasets[0].data.length)
        
    data.forEach( (elt) => {
        addData(chart, chart.data.datasets[0].data.length, elt, 0)
    });

    if (best_model != ""){
        let data = best_data.slice(chart.data.datasets[1].data.length)
        
        data.forEach( (elt) => {
            addData(chart, chart.data.datasets[1].data.length, elt, 1)
        });
    }
   
    // http request for entity
    let req = new XMLHttpRequest();
    console.log("http://localhost:4000/get_acc/" + entity)
    req.open("GET", "http://localhost:4000/get_acc/" + entity);

    req.setRequestHeader("Accept", "application/json");
    req.setRequestHeader("Content-Type", "application/json");
    req.onload = () => {
        let new_data = Function("return " + JSON.parse(req.responseText)["response"] )();

        console.log(entity,":",new_data)
        
        // http request for best model if needed
        if (best_model != "") {
            let req = new XMLHttpRequest();
            console.log("http://localhost:4000/get_acc/" + best_model)
            req.open("GET", "http://localhost:4000/get_acc/" + best_model);
    
            req.setRequestHeader("Accept", "application/json");
            req.setRequestHeader("Content-Type", "application/json");
            req.onload = () => {
                let new_best_data = Function("return " + JSON.parse(req.responseText)["response"] )();
    
                console.log(best_model,":",new_best_data)
                setTimeout(function(){getGlobalAcc(new_data,chart,entity,new_best_data,best_model)},15000);
            
            };
    
            req.send();
        }else{
            setTimeout(function(){getGlobalAcc(new_data,chart,entity, [], "" )},15000);
        }

       
    };

    req.send();
    
}

function createChart(accuracies,id,entity, best_entity,best_accuracies){
    let globalChart;
    let color;
    
    if (id.includes("device")){
        color = 'rgb(138, 99, 255)'
    }else{
        color = 'rgb(255, 99, 132)'
    }
    var labels = []
    for (i=1; i<= accuracies.length; i++){
        labels.push(i);
    }
 
    var data = {
        labels: labels,
        datasets: [{
            label: 'Accuracy',
            backgroundColor: color,
            borderColor: color,
            data: accuracies,
        }
        ]
    };

    // Model has two lines
    if (! id.includes("device")){
        data["datasets"].push (
            {
                label: 'Best accuracy',
                backgroundColor: "rgb(138, 99, 255)",
                borderColor: "rgb(138, 99, 255)",
                data: best_accuracies,
            }
        )
    }

    const config = {
    type: 'line',
    data: data,
    options: {}
    };

    globalChart = new Chart(
        document.getElementById(id),
        config
    );

    getGlobalAcc([], globalChart, entity, [], best_entity)

}

