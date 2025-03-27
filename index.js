const express = require("express");
const app = express();

const PORT = process.env.PORT || 3000;

app.get("/", (req, res) => {
  res.send("<h1>Hello from Railway!</h1><p>You're live 🚀</p>");
});

app.get("/api/hello", (req, res) => {
  res.json({ message: "Hello from the API!" });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});
