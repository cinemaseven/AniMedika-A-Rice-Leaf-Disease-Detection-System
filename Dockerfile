FROM nginx:alpine

COPY home.html /usr/share/nginx/html/index.html
COPY css /usr/share/nginx/html/css
COPY images /usr/share/nginx/html/images
COPY js /usr/share/nginx/html/js
COPY model_web /usr/share/nginx/html/model_web

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]