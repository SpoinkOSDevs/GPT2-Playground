// Import necessary modules
const express = require('express');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');

// Create an instance of Express
const app = express();
const port = 3000;

// Use bodyParser to parse POST request data
app.use(bodyParser.urlencoded({ extended: true }));

// Serve HTML file
app.get('/', (req, res) => {
    res.sendFile(__dirname + 'templates/page.html');
});

// Handle POST request to generate text
app.post('/generate', (req, res) => {
    const prompt = req.body.prompt;

    // Use PythonShell to execute the Python script
    let options = {
        scriptPath: __dirname,
        args: [prompt],
    };

    PythonShell.run('gen.py', options, (err, results) => {
        if (err) throw err;
        
        const generated_text = results[0];
        res.send({ prompt, generated_text });
    });
});

// Listen on the specified port
app.listen(port, () => {
    console.log(`Server is running on port ${port}`);
});
