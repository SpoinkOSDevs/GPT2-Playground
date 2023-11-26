// Import necessary modules
const express = require('express');
const bodyParser = require('body-parser');
const { PythonShell } = require('python-shell');

// Create an Express app
const app = express();
const port = 8080;

// Set up middleware
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

// Serve HTML, CSS, and JS files
app.use(express.static('public'));

// Define a route for text generation
app.post('/generate', (req, res) => {
    const prompt = req.body.prompt;

    // Use PythonShell to run the Python code
    let options = {
        mode: 'text',
        scriptPath: 'start.py', // Adjust the path
        args: [prompt],
    };

    PythonShell.run('start.py', options, (err, results) => {
        if (err) throw err;
        const generatedText = results[0];
        res.send({ generatedText });
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Server is running on http://localhost:${port}`);
});
