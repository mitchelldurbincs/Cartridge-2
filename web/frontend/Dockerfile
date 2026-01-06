# Frontend Service Dockerfile
# Multi-stage build: Node.js build + nginx serve

# ============================================
# Stage 1: Build Frontend (Node.js)
# ============================================
FROM node:20-bookworm-slim AS builder

WORKDIR /app

# Copy package files first for better caching
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy frontend source
COPY . ./

# Build the frontend
RUN npm run build

# ============================================
# Stage 2: Serve with nginx (unprivileged)
# ============================================
FROM nginxinc/nginx-unprivileged:1.27-alpine

# Copy built frontend to nginx html directory
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config (user already set to nginx:nginx)
COPY nginx.conf /etc/nginx/conf.d/default.conf

# nginx-unprivileged runs on port 8080 by default as non-root
EXPOSE 8080

# Healthcheck to verify nginx is serving content
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD wget -q --spider http://localhost:8080/ || exit 1

CMD ["nginx", "-g", "daemon off;"]
