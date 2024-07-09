// user.route.js
import express from 'express';
import { FakeNewsDetection } from '../controller/user.controller.js';

const router = express.Router();

router.post('/detection', FakeNewsDetection);

export default router;
