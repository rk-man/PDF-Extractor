const express = require("express");
const multer = require("multer");
const nlp = require("compromise");
const pdf = require("pdf-parse");
const Word2Vec = require("word2vec");
const use = require("@tensorflow-models/universal-sentence-encoder");
require("@tensorflow/tfjs-core");
require("@tensorflow/tfjs-backend-cpu");
require("@tensorflow/tfjs-node");

const app = express();

app.use(express.json());

const upload = multer();

// Example usage with file upload (assuming Express.js)
app.post("/upload", upload.single("file"), async (req, res) => {
    try {
        const file = req.file;
        if (!file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        let maxWords = 50;
        const text = await pdf(file.buffer);
        const doc = nlp(text.text);
        const sentences = doc.sentences().out("array");
        // console.log(sentences);

        const chunks = [];
        let currentChunk = "";
        for (const sentence of sentences) {
            if ((currentChunk + " " + sentence).split(" ").length <= maxWords) {
                currentChunk += " " + sentence;
            } else {
                chunks.push(currentChunk.trim());
                currentChunk = sentence;
            }
        }

        if (currentChunk.trim() !== "") {
            chunks.push(currentChunk.trim());
        }
        // console.log(chunks);

        const model = await use.load();

        const embeddings = await model.embed(chunks);
        embeddings.print(true);
        console.log(embeddings);

        return res.status(200).json({
            embeddings,
        });
    } catch (error) {
        console.error("Upload and Processing Error:", error);
        return res.status(500).json({ error: "Internal server error" });
    }
});

module.exports = app;
