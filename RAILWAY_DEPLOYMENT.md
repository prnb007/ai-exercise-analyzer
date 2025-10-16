# 🚀 Railway Deployment Guide

## Prerequisites
1. **GitHub Repository**: Your code must be pushed to GitHub
2. **Railway Account**: Sign up at [railway.app](https://railway.app)
3. **MongoDB Atlas**: Set up a MongoDB Atlas cluster

## Step 1: Environment Variables Setup

In Railway Dashboard → Your Project → Variables, set these:

### Required Variables:
```
SECRET_KEY=your-very-secure-secret-key-here
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database_name
FLASK_ENV=production
```

### Optional Variables:
```
FLASK_APP=app_mongodb.py
```

## Step 2: Deploy from GitHub

1. **Connect GitHub**: In Railway Dashboard, click "New Project" → "Deploy from GitHub repo"
2. **Select Repository**: Choose your `ai-exercise-analyzer` repository
3. **Auto-Deploy**: Railway will automatically detect Python and deploy

## Step 3: Verify Deployment

1. **Check Logs**: Monitor deployment logs in Railway Dashboard
2. **Test Health Check**: Visit your Railway URL + `/`
3. **Test Features**: 
   - User registration/login
   - Exercise analysis
   - Video upload/processing

## Step 4: Custom Domain (Optional)

1. **Add Domain**: In Railway Dashboard → Settings → Domains
2. **Configure DNS**: Point your domain to Railway
3. **SSL Certificate**: Railway provides automatic SSL

## Troubleshooting

### Common Issues:

1. **Build Fails**: Check `requirements.txt` has all dependencies
2. **Port Issues**: Ensure app uses `os.environ.get('PORT')`
3. **Database Connection**: Verify `MONGODB_URI` is correct
4. **Memory Issues**: Railway free tier has memory limits

### Performance Optimization:

1. **Gunicorn Workers**: Adjust worker count based on Railway plan
2. **Video Processing**: Consider using external video processing services
3. **Database Indexing**: Add indexes for frequently queried fields

## Production Checklist

- ✅ Environment variables set
- ✅ MongoDB Atlas configured
- ✅ Security headers enabled
- ✅ Production server (Gunicorn) configured
- ✅ Error handling implemented
- ✅ Health check endpoint working
- ✅ File upload limits set
- ✅ CORS configured (if needed)

## Monitoring

- **Logs**: Check Railway Dashboard → Deployments → Logs
- **Metrics**: Monitor CPU, Memory, Network usage
- **Errors**: Set up error tracking (Sentry, etc.)

## Scaling

- **Horizontal**: Add more Railway services
- **Vertical**: Upgrade Railway plan
- **Database**: Use MongoDB Atlas scaling features
- **CDN**: Add CloudFlare or similar for static assets
