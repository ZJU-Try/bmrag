FROM nginx:alpine

# 清除默认配置和默认 server 块（否则会与我们的配置冲突抢占 80 端口）
RUN rm -rf /usr/share/nginx/html/* /etc/nginx/conf.d/default.conf

# 复制静态资源
COPY index.html /usr/share/nginx/html/
COPY css/ /usr/share/nginx/html/css/
COPY js/ /usr/share/nginx/html/js/

# 复制 nginx 配置模板（nginx 官方镜像启动时自动用环境变量替换 ${VAR}）
COPY nginx.conf.template /etc/nginx/templates/nginx.conf.template

# 默认后端地址：指向宿主机（前端单独跑时，宿主机用 python app.py 或 gunicorn 跑后端）
# 联合部署时通过 -e BACKEND_URL=http://backend:5000 覆盖
ENV BACKEND_URL=http://host.docker.internal:5000

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
