const express = require("express");
const uuid = require("uuid");
const multer = require("multer");
const { OpenAI } = require("openai");
const nlp = require("compromise");
const pdf = require("pdf-parse");
const Word2Vec = require("word2vec");
const use = require("@tensorflow-models/universal-sentence-encoder");
require("@tensorflow/tfjs-core");
require("@tensorflow/tfjs-backend-cpu");
require("@tensorflow/tfjs-node");
const { Client } = require("@elastic/elasticsearch");
const client = new Client({
    node: "http://localhost:9200",
});

const openai = new OpenAI({
    apiKey: process.env.CHATGPT_KEY,
});

const app = express();

app.use(express.json());

const upload = multer();

// Example usage with file upload (assuming Express.js)
app.post("/upload", upload.single("file"), async (req, res) => {
    try {
        const file = req.file;
        console.log(file);
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
        const vectors = embeddings.arraySync();
        // embeddings.print(true);
        console.log("CHUNKS");
        console.log(chunks.length, chunks[0].length);
        console.log("EMBEDDINGS");
        console.log(vectors.length, vectors[0].length);

        // CREATE AN INDEX
        const indexName = "vector-embeddings";

        if (!(await client.indices.exists({ index: indexName }))) {
            await client.indices.create({
                index: indexName,
                mappings: {
                    properties: {
                        vectors: {
                            type: "dense_vector",
                            dims: 512,
                            index: true,
                            similarity: "l2_norm",
                        },
                        text: {
                            type: "text",
                        },
                        filename: {
                            type: "text",
                        },
                        doc_id: {
                            type: "text",
                        },
                    },
                },
            });
        }

        const doc_id = uuid.v4();

        const filename = file.originalname.split(".")[0];

        const documents = [];

        for (let idx = 0; idx < chunks.length; idx++) {
            documents.push(
                { index: { _index: indexName } },
                {
                    vectors: vectors[idx],
                    text: chunks[idx],
                    filename: filename,
                    doc_id,
                }
            );
        }

        const bulk_ingestion = await client.bulk({
            operations: documents,
        });
        console.log(bulk_ingestion.items);

        return res.status(200).json({
            status: "success",
        });
    } catch (error) {
        console.error("Upload and Processing Error:", error);
        return res.status(500).json({ error: "Internal server error" });
    }
});

app.get("/documents/:doc_id", async (req, res) => {
    try {
        const { query } = req.query;
        const { doc_id } = req.params;
        const indexName = "vector-embeddings";

        // conver user's query to vector
        const model = await use.load();

        const embeddings = await model.embed(query);
        const vectors = embeddings.arraySync();

        // Define the user's query

        // if (!query) {
        //     return res.status(400).json({
        //         message: "Please enter some query",
        //     });
        // }

        const searchResponse = await client.search({
            index: indexName,
            body: {
                knn: {
                    field: "vectors",
                    query_vector: vectors[0],
                    k: 10, // number of most similar results
                    num_candidates: 100,
                },

                query: {
                    bool: {
                        filter: [
                            {
                                term: {
                                    doc_id,
                                },
                            },
                        ],
                    },
                },
            },
        });

        const searchResults = searchResponse.hits.hits.map((hit) => {
            return hit._source.text;
        });

        const gpt_formatted_search_results = searchResults.map((sr) => {
            return {
                role: "user",
                content: sr,
            };
        });

        // open ai chat completion
        const chatgpt_res = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                {
                    role: "system",
                    content:
                        "You are someone who explains things in a simple and clear manner with the help of some sentences",
                },
                ...gpt_formatted_search_results,
                {
                    role: "user",
                    content: query,
                },
            ],
        });

        return res.status(200).json({
            status: "success",
            results: chatgpt_res.choices[0].message.content,
        });
    } catch (err) {
        console.log(err);
        return res.status(500).json({ error: "Internal server error" });
    }
});

module.exports = app;
