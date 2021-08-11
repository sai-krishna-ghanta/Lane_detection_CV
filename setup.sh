mkdir -p ~/.streamlit/
echo "\
[general]\n\
email = \"your-email@domain.com\"\n\
" > ~/.streamlit/credentials.toml
# adding lib symlink
ln -s $BUILD_DIR/vendor/libreoffice/deps/lib/libGL.so.8.0.0 
$BUILD_DIR/vendor/libreoffice/deps/lib/libGL.so.1

echo "\
[server]\n\
headless = true\n\
enableCORS=false\n\
port = $PORT\n\
" > ~/.streamlit/config.toml
