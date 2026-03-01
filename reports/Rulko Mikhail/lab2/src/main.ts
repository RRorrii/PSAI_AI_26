const dataset: DataPoint[] = [
    { x1: 1,  x2: 4,  label: -1 }, 
    { x1: -1, x2: 4,  label: -1 },
    { x1: 1,  x2: -4, label: -1 },
    { x1: -1, x2: -4, label: 1 }  
];

const viz = new Visualizer('mseChart', 'decisionChart');

const netFixed = new Perceptron();
const resFixed = netFixed.train(dataset, 'fixed', 0.03); 

const netAdapt = new Perceptron();
const resAdapt = netAdapt.train(dataset, 'adaptive');

viz.drawLearningCurves(resFixed.history, resAdapt.history);
viz.drawBoundary(netAdapt, dataset);