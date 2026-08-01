# ====== 构建阶段：Node 编译 React ======
FROM node:20-alpine AS builder

WORKDIR /app

# 先复制 package.json，利用层缓存
COPY package.json ./

# 安装依赖
RUN npm config set registry https://registry.npmmirror.com && npm install

# 复制源码并构建
COPY . .
RUN npm run build

# ====== 运行阶段：nginx 部署 ======
FROM nginx:alpine

# 清除默认配置
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf

# 从 builder 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

# 复制 nginx 配置模板
COPY nginx.conf.template /etc/nginx/templates/nginx.conf.template

# 默认后端地址：指向宿主机
ENV BACKEND_URL=http://host.docker.internal:5000

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
