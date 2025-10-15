# 🚀 Railway Deployment Guide

## Prerequisites
- GitHub repository: [https://github.com/prnb007/ai-exercise-analyzer](https://github.com/prnb007/ai-exercise-analyzer)
- MongoDB Atlas account
- Railway account

## Step 1: Deploy on Railway

1. **Go to [Railway.app](https://railway.app)** and sign in
2. **Click "New Project"**
3. **Select "Deploy from GitHub repo"**
4. **Choose your repository** (`prnb007/ai-exercise-analyzer`)
5. **Railway will automatically detect** it's a Python project

## Step 2: Configure Environment Variables

In Railway dashboard, go to your project → Variables tab and add:

```bash
# MongoDB Configuration
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/exercise_analyzer?retryWrites=true&w=majority

# Flask Configuration
SECRET_KEY=your-super-secret-key-here
FLASK_ENV=production

# Optional: Custom port (Railway will set this automatically)
PORT=5000
```

## Step 3: MongoDB Atlas Setup

1. **Go to [MongoDB Atlas](https://cloud.mongodb.com)**
2. **Create a new cluster** (free tier available)
3. **Create database user** with read/write permissions
4. **Whitelist Railway's IP ranges** (0.0.0.0/0 for development)
5. **Get connection string** and update `MONGODB_URI`

## Step 4: Deploy

1. **Railway will automatically build and deploy** your application
2. **Check the logs** for any errors
3. **Your app will be available** at the provided Railway URL

## Step 5: Custom Domain (Optional)

1. **Go to Railway project settings**
2. **Add custom domain** if you have one
3. **Configure DNS** as instructed

## Troubleshooting

### Common Issues:
- **MediaPipe errors**: These are expected in production and won't affect functionality
- **MongoDB connection**: Ensure your connection string is correct
- **Port issues**: Railway handles this automatically

### Performance Optimization:
- **Enable caching** for static files
- **Use CDN** for better performance
- **Monitor resource usage** in Railway dashboard

## Production Checklist

- ✅ Environment variables configured
- ✅ MongoDB Atlas connected
- ✅ Static files served correctly
- ✅ Error handling in place
- ✅ Security headers configured
- ✅ HTTPS enabled (Railway default)

## Support

- **Railway Documentation**: [https://docs.railway.app](https://docs.railway.app)
- **MongoDB Atlas**: [https://docs.atlas.mongodb.com](https://docs.atlas.mongodb.com)
- **Flask Deployment**: [https://flask.palletsprojects.com/en/2.0.x/deploying/](https://flask.palletsprojects.com/en/2.0.x/deploying/)

---

**Your app will be live at**: `https://your-app-name.railway.app`
