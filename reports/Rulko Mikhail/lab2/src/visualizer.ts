class Visualizer {
    private mseCanvas: HTMLCanvasElement;
    private decisionCanvas: HTMLCanvasElement;
    private ctxMse: CanvasRenderingContext2D;
    private ctxDecision: CanvasRenderingContext2D;

    constructor(mseId: string, decisionId: string) {
        this.mseCanvas = document.getElementById(mseId) as HTMLCanvasElement;
        this.decisionCanvas = document.getElementById(decisionId) as HTMLCanvasElement;
        
        this.ctxMse = this.mseCanvas.getContext('2d')!;
        this.ctxDecision = this.decisionCanvas.getContext('2d')!;
    }

    public drawLearningCurves(fixedHistory: number[], adaptiveHistory: number[]): void {
        const ctx = this.ctxMse;
        const { width: w, height: h } = this.mseCanvas;
        ctx.clearRect(0, 0, w, h);

        const maxLen = Math.max(fixedHistory.length, adaptiveHistory.length);
        const maxMse = Math.max(fixedHistory[1] || 1, adaptiveHistory[1] || 1);

        const drawPath = (data: number[], color: string, label: string, yOffset: number) => {
            ctx.beginPath();
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            data.forEach((val, i) => {
                const x = (i / maxLen) * (w - 60) + 40;
                const y = h - (Math.min(val, maxMse) / maxMse) * (h - 80) - 40;
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.fillStyle = color;
            ctx.fillText(`${label}: ${data.length} epochs`, 40, yOffset);
        };

        drawPath(fixedHistory, '#e74c3c', 'Fixed', 20);
        drawPath(adaptiveHistory, '#2ecc71', 'Adaptive', 40);
    }

    public drawBoundary(model: Perceptron, dataset: DataPoint[]): void {
        const ctx = this.ctxDecision;
        const { width: w, height: h } = this.decisionCanvas;
        const scale = 35;
        const center = { x: w / 2, y: h / 2 };

        ctx.clearRect(0, 0, w, h);

        ctx.strokeStyle = '#eee';
        ctx.beginPath();
        ctx.moveTo(0, center.y); ctx.lineTo(w, center.y);
        ctx.moveTo(center.x, 0); ctx.lineTo(center.x, h);
        ctx.stroke();

        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (let x1 = -10; x1 <= 10; x1 += 0.5) {
            const x2 = (-model.bias - model.w1 * x1) / model.w2;
            const cx = center.x + x1 * scale;
            const cy = center.y - x2 * scale;
            if (x1 === -10) ctx.moveTo(cx, cy);
            else ctx.lineTo(cx, cy);
        }
        ctx.stroke();

        dataset.forEach(p => {
            ctx.fillStyle = p.label === 1 ? '#2ecc71' : '#e67e22';
            ctx.beginPath();
            ctx.arc(center.x + p.x1 * scale, center.y - p.x2 * scale, 5, 0, Math.PI * 2);
            ctx.fill();
        });
    }
}