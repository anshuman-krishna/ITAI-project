import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // standalone output keeps the production docker image small
  output: "standalone",
  // @itai/shared ships typescript source, so let next transpile it
  transpilePackages: ["@itai/shared"],
};

export default nextConfig;
