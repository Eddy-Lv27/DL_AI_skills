import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 模块四 B1：dev server 把 /api 转发到本地后端（uvicorn :8000），
    // 生产构建可用 VITE_API_BASE 覆盖 API 基址。
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
})
