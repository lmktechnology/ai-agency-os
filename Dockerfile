# Use official n8n image
FROM n8nio/n8n:1.111.0

# Work directory
WORKDIR /data

# Expose port
EXPOSE 5678

# Mount persistent storage
VOLUME /data

