class Perceptron {
    public w1: number = 0;
    public w2: number = 0;
    public bias: number = 0;
    private readonly epsilon: number = 0.001;

    public getSum(x1: number, x2: number): number {
        return x1 * this.w1 + x2 * this.w2 + this.bias;
    }

    public predict(x1: number, x2: number): number {
        return this.getSum(x1, x2) >= 0 ? 1 : -1;
    }

    public train(
        dataset: DataPoint[], 
        mode: TrainingMode = 'fixed', 
        alphaFixed: number = 0.01
    ): TrainingResult {
        let epochs = 0;
        const history: number[] = [];
        const maxEpochs = 10;

        while (epochs < maxEpochs) {
            let mseSum = 0;
            const shuffled = [...dataset].sort(() => Math.random() - 0.5);

            shuffled.forEach(point => {
                const out = this.getSum(point.x1, point.x2);
                const error = point.label - out;

                let alpha = alphaFixed;
                if (mode === 'adaptive') {
                    const norm = 1 + Math.pow(point.x1, 2) + Math.pow(point.x2, 2);
                    alpha = 0.5 * (1 / norm);
                }

                this.w1 += alpha * error * point.x1;
                this.w2 += alpha * error * point.x2;
                this.bias += alpha * error;

                mseSum += Math.pow(error, 2);
            });

            const avgMse = mseSum / dataset.length;
            history.push(avgMse);
            epochs++;

            if (avgMse <= this.epsilon) break;
        }
        return { history, epochs };
    }
}