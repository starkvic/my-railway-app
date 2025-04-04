const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static files (CSS, images, JS)
app.use(express.static(path.join(__dirname, "public")));

// Routes
app.get("/", (req, res) => res.sendFile(path.join(__dirname, "views/index.html")));
app.get("/about", (req, res) => res.sendFile(path.join(__dirname, "views/about.html")));
app.get("/simulation", (req, res) => res.sendFile(path.join(__dirname, "views/simulation.html")));
app.get("/results", (req, res) => res.sendFile(path.join(__dirname, "views/results.html")));
app.get("/interactive", (req, res) => {res.render("interactive");});

// Start server
app.listen(PORT, () => {
  console.log(`Server is running at http://localhost:${PORT}`);
});
