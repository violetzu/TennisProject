/** @type {import('next').NextConfig} */

const allowedDevOrigins = process.env.NEXT_ALLOWED_DEV_ORIGINS
  ? process.env.NEXT_ALLOWED_DEV_ORIGINS.split(',').map(s => s.trim())
  : []

const backend_domain = process.env.BACKEND_DOMAIN || "http://backend:8000";

const nextConfig = {
  allowedDevOrigins,

  experimental: {
    proxyClientMaxBodySize: '200mb',
  },

  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `${backend_domain}/api/:path*`,
      },
      {
        source: '/videos/:path*',
        destination: `${backend_domain}/videos/:path*`,
      },
      {
        source: '/docs',
        destination: `${backend_domain}/docs`,
      },
      {
        source: '/openapi.json',
        destination: `${backend_domain}/openapi.json`,
      },
    ]
  },
}

module.exports = nextConfig
