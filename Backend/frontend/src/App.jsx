import React, { useState } from "react";

function App() {
  const [newsText, setNewsText] = useState("");
  const [result, setResult] = useState(null);

  const handleChange = (e) => {
    setNewsText(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    // Make API call to backend to get the prediction
    try {
      const response = await fetch("/api/user/detection", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text: newsText }),
      });
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error("Error fetching prediction:", error);
      setResult({ prediction: "Error", probability: 0 });
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex flex-col justify-center items-center py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-md w-full space-y-8">
      <div className="bg-blue-400 text-white py-4 px-6">
      <h1 className="text-3xl font-bold">Welcome to MisInfoSlayer</h1>
    </div>
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">Fake News Detector</h2>
        </div>
        <form className="mt-8 space-y-6" onSubmit={handleSubmit}>
          <textarea
            value={newsText}
            onChange={handleChange}
            className="shadow-sm focus:ring-indigo-500 focus:border-indigo-500 block w-full sm:text-sm border-gray-300 rounded-md p-2 bg-white"
            placeholder="Enter news text here"
            rows="6"
          ></textarea>
          <div className="flex justify-center">
            <button
              type="submit"
              className="mt-4 w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              Predict
            </button>
          </div>
        </form>
        {result && (
          <div className="mt-8">
            <h3 className="text-lg leading-6 font-medium text-gray-900">Result:</h3>
            <div className={`mt-2 px-4 py-2 rounded-lg text-center ${result.prediction === "FAKE" ? "bg-red-200 text-red-800" : "bg-green-200 text-green-800"}`}>
              <p>Prediction: {result.prediction}</p>
              <p>Probability: {result.probability.toFixed(2)}</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
