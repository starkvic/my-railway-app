const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, "public")));

app.get("/", (req, res) => res.sendFile(path.join(__dirname, "views/index.html")));
app.get("/about", (req, res) => res.sendFile(path.join(__dirname, "views/about.html")));
app.get("/simulation", (req, res) => res.sendFile(path.join(__dirname, "views/simulation.html")));

app.listen(PORT, () => console.log(`App running on port ${PORT}`));
