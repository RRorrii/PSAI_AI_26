interface DataPoint {
    x1: number;
    x2: number;
    label: 1 | -1;
}

type TrainingMode = 'fixed' | 'adaptive';

interface TrainingResult {
    history: number[];
    epochs: number;
}