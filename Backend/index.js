import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import path from "path";
import bodyParser from "body-parser";
import cookieParser from "cookie-parser"; // Import cookie-parser

// Load environment variables
dotenv.config();

const app = express(); // Define the app instance
const PORT = process.env.PORT || 3001;
const URI = process.env.MongoDBURI;

// Middleware
app.use(cors());
app.use(cookieParser()); // Use cookie-parser middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(bodyParser.json());

// Define routes
// Assuming userRoute is imported correctly
import userRoute from "./routes/user.route.js"; // Update the import as necessary
app.use("/api/user", userRoute);

// Code for deployment
if (process.env.NODE_ENV === "production") {
    const dirPath = path.resolve();
    app.use(express.static(path.join(dirPath, "frontend", "dist")));
    app.get("*", (req, res) => {
        res.sendFile(path.resolve(dirPath, "frontend", "dist", "index.html"));
    });
}

// Start the server
app.listen(PORT, () => {
    console.log(`Server is listening on port ${PORT}`);
});
