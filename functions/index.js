const functions = require('firebase-functions');
const { createApp } = require('./main');

const app = createApp();

exports.api = functions.https.onRequest(app);