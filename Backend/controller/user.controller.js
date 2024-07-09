import { spawn } from "child_process";
import { fileURLToPath } from 'url';
import path from "path";

export const FakeNewsDetection = (req, res) => {
    const newsText = req.body.text;
    const scriptUrl = new URL('./detector.py', import.meta.url);
    const scriptPath = fileURLToPath(scriptUrl);

    const pythonProcess = spawn('python', [scriptPath, newsText]);

    pythonProcess.stdout.on('data', (data) => {
        const result = JSON.parse(data.toString());
        res.status(200).json(result);
    });

    pythonProcess.stderr.on('data', (data) => {
        console.error(`stderr: ${data}`);
        res.status(500).send('Error in detecting fake news.');
    });

    pythonProcess.on('close', (code) => {
        console.log(`child process exited with code ${code}`);
    });
};
