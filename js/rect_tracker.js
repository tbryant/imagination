var data, labels;
var layer_defs, net, trainer;

// create neural net
var layer_defs = [];
layer_defs.push({
    type: 'input',
    out_sx: 16,
    out_sy: 16,
    out_depth: 2
});
// 2 inputs: x, y
layer_defs.push({
    type: 'fc',
    num_neurons: 20,
    activation: 'relu'
});
layer_defs.push({
    type: 'regression',
    num_neurons: 2
});
// 3 outputs: r,g,b

var net = new convnetjs.Net();
net.makeLayers(layer_defs);

var x = new convnetjs.Vol([]);

var trainer = new convnetjs.SGDTrainer(net,{
    learning_rate: 0.01,
    momentum: 0.9,
    batch_size: 10,
    l2_decay: 0.0
});

var train = true;

var currentNet;

var batches_per_iteration = 1;
var mod_skip_draw = 1;
var smooth_loss = -1;

function update() {
    if (!train)
        return;
    // forward prop the data
    var W = nn_canvas.width;
    var H = nn_canvas.height;
    
    var p = ori_data.data;
    
    currentNet = new convnetjs.Vol(16,16,2);
    var loss = 0;
    var lossi = 0;
    var N = batches_per_iteration;
    for (var iters = 0; iters < trainer.batch_size; iters++) {
        for (var i = 0; i < N; i++) {
            // sample all coordinate
            for (var x = 0; x < 16; x++) {
                for (var y = 0; y < 16; y++) {
                    var ix = ((W * y) + x) * 4;
                    var value = p[ix] / 255.0 + p[ix + 1] / 255.0 + p[ix + 2] / 255.0;
                    // overall brightness - perhaps hue?
                    currentNet.w[ix / 4] = value;
                }
            }
            
            var stats = trainer.train(currentNet, ori_data.params);
            loss += stats.loss;
            lossi += 1;
        
        }
    }
    loss /= lossi;
    
    if (counter === 0)
        smooth_loss = loss;
    else
        smooth_loss = 0.99 * smooth_loss + 0.01 * loss;
    
    var t = '';
    t += 'loss: ' + smooth_loss;
    t += '<br>'
    t += 'iteration: ' + counter;
    document.getElementById("report").innerHTML = t;
}

function draw() {
    params = net.forward(currentNet).w;
    render(nn_ctx, params);
}

function render(ctx, params) {
    //draw a rect in the middle-ish
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, sz, sz);
    ctx.fillStyle = "red";
    ctx.fillRect(Math.round(params[0] * sz), Math.round(params[1] * sz), 2, 2);

}

// evaluate current network on test set
var testPredict = function() {
    
    var params = [Math.random(), Math.random()];
    render(test_input_ctx, params);
    
    var v = new convnetjs.Vol(16,16,2);

    var p = test_input_ctx.getImageData(0, 0, sz, sz).data;
    // sample all coordinate
    for (var x = 0; x < 16; x++) {
        for (var y = 0; y < 16; y++) {
            var ix = ((sz * y) + x) * 4;
            var value = p[ix] / 255.0 + p[ix + 1] / 255.0 + p[ix + 2] / 255.0;
            // overall brightness - perhaps hue?
            v.w[ix / 4] = value;
        }
    }
    
    params = net.forward(v).w;
//     console.log(params);
    
    render(test_output_ctx, params);

}


function tick() {
    update();
    draw();
    // run prediction on test set
    if ((counter % 100 === 0 && counter > 0) || counter === 100) {
        testPredict();
    }
    counter += 1;
}


function moveRect() {
    var params = [Math.random(), Math.random()];
    render(ori_ctx, params);
    ori_data = ori_ctx.getImageData(0, 0, sz, sz);
    ori_data.params = params;
}



var ori_canvas, nn_canvas, ori_ctx, nn_ctx, ori_data;
var test_input_canvas, test_input_ctx;
var test_output_canvas, test_output_ctx;

var sz = 16;
// size of our drawing area
var counter = 0;

document.addEventListener("DOMContentLoaded", function(event) {
    ori_canvas = document.getElementById('canvas_original');
    nn_canvas = document.getElementById('canvas_net');
    ori_canvas.width = sz;
    ori_canvas.height = sz;
    nn_canvas.width = sz;
    nn_canvas.height = sz;
    
    test_input_canvas = document.getElementById('canvas_test_input');
    test_input_canvas.width = sz;
    test_input_canvas.height = sz;
    test_input_ctx = test_input_canvas.getContext("2d");
    
    test_output_canvas = document.getElementById('canvas_test_output');
    test_output_canvas.width = sz;
    test_output_canvas.height = sz;
    test_output_ctx = test_output_canvas.getContext("2d");
    
    
    var params = [0.5, 0.5];
    ori_ctx = ori_canvas.getContext("2d");
    render(ori_ctx, params);
    moveRect();
    
    nn_ctx = nn_canvas.getContext("2d");
    
    //periodically update the input
    setInterval(moveRect, 1);
    
    // start the regression!
    setInterval(tick, 1);

});
