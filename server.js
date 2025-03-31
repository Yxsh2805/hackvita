require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { ApifyClient } = require('apify-client');
const fs = require('fs');
const { Parser } = require('json2csv');
const path = require('path');

const app = express();
const PORT = 3001;

app.use(cors({
  origin: ['http://localhost:3000', 'http://127.0.0.1:3000'], 
  methods: ['POST', 'GET']
}));

app.use(express.json());

// Ensure APIFY_TOKEN is set
if (!process.env.APIFY_TOKEN) {
  console.error('❌ FATAL ERROR: Missing APIFY_TOKEN in .env file');
  process.exit(1);
}

let client;
try {
  client = new ApifyClient({ token: process.env.APIFY_TOKEN });
  console.log('✅ Apify client initialized successfully');
} catch (err) {
  console.error('❌ Failed to initialize Apify client:', err);
  process.exit(1);
}

// ✅ Scrape Instagram Comments
app.post('/api/scrape-comments', async (req, res) => {
  try {
    let { postUrl, resultsLimit = 20 } = req.body;

    if (!postUrl || typeof postUrl !== 'string') {
      return res.status(400).json({ error: "Valid Instagram post URL is required" });
    }

    if (!/^https:\/\/(www\.)?instagram\.com\/p\/.+/.test(postUrl)) {
      postUrl = `https://www.instagram.com/p/${postUrl.replace(/.*instagram\.com\/p\/|\/$/, '')}/`;
    }

    const input = {
      directUrls: [postUrl],
      resultsType: 'comments',
      resultsLimit: Math.min(Math.max(Number(resultsLimit), 1, 100)),
      proxyConfiguration: { useApifyProxy: true }
    };

    console.log('✅ Scraping comments for:', postUrl);

    const run = await client.actor("shu8hvrXbJbY3Eb9W").call(input);
    const { items } = await client.dataset(run.defaultDatasetId).listItems();

    if (!items.length) {
      return res.status(404).json({ error: "No comments found for this post" });
    }

    const comments = items.map(comment => ({
      username: comment.username || 'Unknown',
      text: comment.text || 'No text',
      timestamp: comment.timestamp || new Date().toISOString()
    }));

    // Call CSV export function
    const csvFilePath = exportCommentsToCSV(comments);

    res.json({ success: true, message: "Comments scraped successfully", comments, csvFilePath });

  } catch (error) {
    console.error('❌ API Error:', error);
    res.status(500).json({ error: "Scraping failed", details: error.message });
  }
});

// ✅ Export Scraped Comments to CSV
function exportCommentsToCSV(comments) {
  try {
    if (!Array.isArray(comments) || comments.length === 0) {
      return null;
    }

    // Format comments: Only include `id` and `text`
    const formattedComments = comments.map((comment, index) => ({
      id: index + 1,  // Assign an auto-incrementing ID
      text: comment.text || 'No text'
    }));

    // Convert to CSV format
    const parser = new Parser({ fields: ['id', 'text'] });
    const csv = parser.parse(formattedComments);

    // Define file path in the "data" directory
    const dirPath = path.join(__dirname, 'data');
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }

    const fileName = `comments_${Date.now()}.csv`;
    const filePath = path.join(dirPath, fileName);
    
    // Write CSV file
    fs.writeFileSync(filePath, csv);
    
    console.log(`✅ CSV saved successfully: ${filePath}`);
    return filePath;

  } catch (error) {
    console.error('❌ CSV Export Error:', error);
    return null;
  }
}

// ✅ Download the CSV File
app.get('/api/download-comments', (req, res) => {
  const dirPath = path.join(__dirname, 'data');

  // Find the latest CSV file
  fs.readdir(dirPath, (err, files) => {
    if (err || files.length === 0) {
      return res.status(404).json({ error: "No CSV files found" });
    }

    const latestFile = files
      .filter(f => f.startsWith('comments_'))
      .sort((a, b) => fs.statSync(path.join(dirPath, b)).mtime - fs.statSync(path.join(dirPath, a)).mtime)[0];

    const filePath = path.join(dirPath, latestFile);
    res.download(filePath);
  });
});

// ✅ Start Server
app.listen(PORT, () => {
  console.log(`🚀 Server running on http://localhost:${PORT}`);
});
