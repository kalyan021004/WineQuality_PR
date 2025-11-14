const express = require("express");
const path = require("path");
const fs = require("fs");

const app = express();

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

app.use(express.static("public"));
app.use("/results", express.static("results")); // serve images

// List of algorithms
const algorithms = [
    { id: "xgb", name: "XGBoost" },
    { id: "lgbm", name: "LightGBM" },
    { id: "gb", name: "Gradient Boosting" },
    { id: "rf", name: "Random Forest" },
    { id: "svm", name: "Support Vector Machine" },
    { id: "catboost", name: "CatBoost" }
];

// Homepage
app.get("/", (req, res) => {
    res.render("index", { algorithms });
});

app.get("/algorithm/:id", (req, res) => {
    try {
        const { id } = req.params;

        const algo = algorithms.find(a => a.id === id);
        if (!algo) return res.send("Invalid algorithm.");

        const metricsPath = path.join(__dirname, "models", `${id}_metrics.json`);
        console.log("Metrics File:", metricsPath);

        const metrics = JSON.parse(fs.readFileSync(metricsPath));

        return res.render("algorithm", {
            algo,
            metrics,
            confusionImage: `/results/${id}_confusion_matrix.png`,
            rocImage: `/results/${id}_roc_curve.png`
        });
    } catch (e) {
        console.log("EJS ERROR:", e);
        res.send(`<h2>Error rendering page:</h2><pre>${e}</pre>`);
    }
});


const PORT = process.env.PORT || 3000;

app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
