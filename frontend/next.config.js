/** @type {import('next').NextConfig} */

const allowedDevOrigins = process.env.NEXT_ALLOWED_DEV_ORIGINS
  ? process.env.NEXT_ALLOWED_DEV_ORIGINS.split(',').map(s => s.trim())
  : []

const nextConfig = {
  allowedDevOrigins,
  
  experimental: {
    proxyClientMaxBodySize: '200mb', 
  },

  async rewrites() {
    return [
      { source: '/api/:path*', destination: 'http://backend:8000/:path*' },
      { source: '/videos/:path*', destination: 'http://backend:8000/videos/:path*' },
      { source: '/docs', destination: 'http://backend:8000/docs' },
      { source: '/openapi.json', destination: 'http://backend:8000/openapi.json' },
    ]
  },
}

module.exports = nextConfig
