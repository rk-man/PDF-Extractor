const express = require("express");
const uuid = require("uuid");
const multer = require("multer");
const { OpenAI } = require("openai");
const pdf = require("pdf-parse");
const { RecursiveCharacterTextSplitter } = require("langchain/text_splitter");
const { Document } = require("langchain/document");
const {
    findDataTypes,
    convertStringToRespectiveTypes,
} = require("./findDataTypes");

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
app.post("/upload/pdf", upload.single("file"), async (req, res) => {
    try {
        const file = req.file;
        console.log(file);
        if (!file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        const text = await pdf(file.buffer);
        let chunks = [];

        const splitter = new RecursiveCharacterTextSplitter({
            chunkSize: 600,
            chunkOverlap: 100,
        });

        const docOutput = await splitter.splitDocuments([
            new Document({ pageContent: text.text.toString() }),
        ]);

        chunks = docOutput.map((doc) => {
            return doc.pageContent;
        });

        // CREATE VECTORS
        const gpt_response = await openai.embeddings.create({
            input: chunks,
            model: "text-embedding-ada-002",
        });

        const vectors = gpt_response.data.map((doc) => {
            return doc.embedding;
        });

        console.log(vectors[0].length);

        // CREATE AN INDEX
        const indexName = "vector-embeddings";

        if (!(await client.indices.exists({ index: indexName }))) {
            await client.indices.create({
                index: indexName,
                mappings: {
                    properties: {
                        vectors: {
                            type: "dense_vector",
                            dims: 1536,
                            index: true,
                            similarity: "dot_product",
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
        //
        const doc_id = uuid.v4();

        const filename = file.originalname.split(".")[0];

        const documents = [];

        for (let idx = 0; idx < chunks.length; idx++) {
            documents.push(
                { index: { _index: indexName } }, // index properties
                {
                    // actual data
                    vectors: vectors[idx],
                    text: chunks[idx],
                    filename: filename,
                    doc_id,
                }
            );
        }

        // ingesting multiple docs all at once.
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

app.get("/pdf/documents/:doc_id", async (req, res) => {
    try {
        const { query } = req.query;
        const { doc_id } = req.params;
        const indexName = "vector-embeddings";

        // CREATE VECTORS

        const gpt_response = await openai.embeddings.create({
            input: query,
            model: "text-embedding-ada-002",
        });

        const vectors = gpt_response.data[0].embedding;
        console.log(vectors);

        // Define the user's query

        if (!query) {
            return res.status(400).json({
                message: "Please enter some query",
            });
        }

        const searchResponse = await client.search({
            index: indexName,
            body: {
                knn: {
                    field: "vectors",
                    query_vector: vectors,
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

        let gpt_formatted_search_results = searchResults.join(" ");
        gpt_formatted_search_results +=
            " Analyze this context, deeply understand it in your own way and answer the following query : " +
            query;

        // open ai chat completion
        const chatgpt_res = await openai.chat.completions.create({
            model: "gpt-3.5-turbo",
            messages: [
                {
                    role: "system",
                    content:
                        "You are an helpful assistant who specializes in understanding a content and have the ability to answer questions in a meaningful way",
                },
                {
                    role: "user",
                    content: gpt_formatted_search_results,
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

app.post("/upload/csv", upload.single("file"), async (req, res, next) => {
    try {
        const file = req.file;
        if (!file) {
            return res.status(400).json({ error: "No file uploaded" });
        }

        // Converting csv buffer to string
        let csv_string = file.buffer.toString();

        // splitting the string into rows
        let csv_arrays = csv_string.split("\n");

        // Getting all the columns
        let csv_columns = csv_arrays[0].split(",").map((item) => {
            return item.trim();
        });

        // getting all data (rows)
        let doc_id = uuid.v4();
        let csv_docs = [];
        for (let i = 1; i < csv_arrays.length; i++) {
            let cur_row = csv_arrays[i].split(",").map((item) => {
                return item.trim();
            });

            let cur_row_obj = {};

            for (let i = 0; i < csv_columns.length; i++) {
                cur_row_obj[`${csv_columns[i]}`] =
                    convertStringToRespectiveTypes(cur_row[i]);
            }
            cur_row_obj.doc_id = i;
            csv_docs.push(cur_row_obj);
        }

        // CREATING A UNIQUE INDEX FOR EACH CSV FILE
        const indexName = file.originalname.split(".")[0].trim();

        if (await client.indices.exists({ index: indexName })) {
            return res.status(400).json({
                message: "Index with this name already exists",
            });
        }

        // INDEX MAPPING SCHEMA PROPERTIES
        const mapping_properties = {};
        for (let col of csv_columns) {
            mapping_properties[`${col}`] = {
                type: findDataTypes(csv_docs[0][col]),
            };
        }
        mapping_properties.doc_id = {
            type: "integer",
        };

        // CREATING INDEX
        await client.indices.create({
            index: indexName,
            mappings: {
                properties: mapping_properties,
            },
        });

        // CREATING DOCUMENTS
        const documents = [];
        for (let doc of csv_docs) {
            documents.push({ index: { _index: indexName } }, doc);
        }

        // BULK INGESTING DOCS
        const bulk_ingestion = await client.bulk({
            operations: documents,
        });

        return res.status(200).json({
            docs: bulk_ingestion.items,
        });
    } catch (err) {
        console.log(err);
        return res.status(500).json({ error: "Internal server error" });
    }
});

app.get("/csv/documents/:doc_id", async (req, res) => {
    try {
        const { query } = req.query;
        const { doc_id } = req.params;
        if (!doc_id) {
            return res.status(400).json({
                status: "fail",
                message: "PLease specify the document id",
            });
        }
        if (!query)
            return res.status(400).json({
                status: "fail",
                message: "PLease enter some sql query",
            });

        const sql_response = await client.sql.query({
            format: "txt",
            query: query,
        });

        return res.status(200).json({
            status: "Success",
            results: sql_response,
        });
    } catch (err) {
        console.log(err);
        return res.status(500).json({ error: "Internal server error" });
    }
});

module.exports = app;
