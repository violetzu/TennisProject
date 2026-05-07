import type { NextConfig } from "next";

const securityHeaders = [
  { key: "X-Frame-Options", value: "DENY" },
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
  { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
];

const backendDomain = process.env.BACKEND_DOMAIN || "http://backend:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  ...(process.env.ALLOWED_DEV_ORIGINS && {
    allowedDevOrigins: process.env.ALLOWED_DEV_ORIGINS.split(",").map((s) => s.trim()),
  }),

  experimental: {
    proxyClientMaxBodySize: "200mb",
  },

  async headers() {
    if (process.env.NODE_ENV !== "production") return [];
    return [{ source: "/(.*)", headers: securityHeaders }];
  },

  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${backendDomain}/api/:path*`,
      },
      {
        source: "/videos/:path*",
        destination: `${backendDomain}/videos/:path*`,
      },
      {
        source: "/docs",
        destination: `${backendDomain}/docs`,
      },
      {
        source: "/openapi.json",
        destination: `${backendDomain}/openapi.json`,
      },
    ];
  },
};

export default nextConfig;
