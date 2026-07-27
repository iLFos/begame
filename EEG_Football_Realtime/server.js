/**
 * =====================================================
 *  本地信号服务器 - 桥接 movda.txt 与游戏页面
 * =====================================================
 *
 * 功能：
 *  1. 持续监听 movda.txt 文件变化
 *  2. 将文件最后一个数字(0/1)通过 HTTP 接口提供给游戏
 *  3. 同时托管 demov1.0.html 供浏览器访问
 *
 * 使用方式：
 *  node server.js
 *
 * 然后浏览器打开 http://localhost:8080
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

// ===== 配置 =====
const PORT = 8080;
const SIGNAL_FILE = path.join(__dirname, 'movda.txt');
const HTML_FILE = path.join(__dirname, 'demov1.0.html');

// ===== 信号状态 =====
let currentSignal = 0;
let lastContent = '';

// ===== 读取并解析信号文件 =====
function readSignalFile() {
  try {
    if (!fs.existsSync(SIGNAL_FILE)) {
      console.log(`[警告] 信号文件不存在: ${SIGNAL_FILE}`);
      console.log(`       请创建该文件并写入 0 或 1`);
      return;
    }

    const content = fs.readFileSync(SIGNAL_FILE, 'utf8').trim();

    // 只在内容变化时打印
    if (content !== lastContent) {
      lastContent = content;
      const lastChar = content.charAt(content.length - 1);
      const val = parseInt(lastChar, 10);

      if (!isNaN(val) && (val === 0 || val === 1)) {
        currentSignal = val;
        const timestamp = new Date().toLocaleTimeString('zh-CN', { hour12: false });
        console.log(`[${timestamp}] 信号更新: ${val === 1 ? '▶ 按下(Z键)' : '○ 松开'}`);
      } else {
        console.log(`[${timestamp}] 文件内容: "${content}" → 无法解析为 0/1, 忽略`);
      }
    }
  } catch (err) {
    console.error(`[错误] 读取信号文件失败: ${err.message}`);
  }
}

// 启动时读一次
readSignalFile();

// 每 30ms 轮询文件变化（比游戏轮询更快，确保实时性）
setInterval(readSignalFile, 30);

// 同时用 fs.watch 监听文件变化（双保险）
try {
  fs.watch(SIGNAL_FILE, (eventType) => {
    if (eventType === 'change') {
      readSignalFile();
    }
  });
  console.log(`[已启动] 文件监听: ${SIGNAL_FILE}`);
} catch (err) {
  console.log(`[提示] fs.watch 不可用，将仅使用轮询模式 (正常)`);
}

// ===== HTTP 服务器 =====
const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js':   'text/javascript; charset=utf-8',
  '.css':  'text/css; charset=utf-8',
  '.txt':  'text/plain; charset=utf-8'
};

const server = http.createServer((req, res) => {
  // 路由处理
  if (req.url === '/signal' || req.url.startsWith('/signal?')) {
    // 信号接口 → 返回当前信号值
    res.writeHead(200, {
      'Content-Type': 'text/plain',
      'Cache-Control': 'no-store, no-cache, must-revalidate',
      'Access-Control-Allow-Origin': '*'
    });
    res.end(String(currentSignal));
    return;
  }

  // 默认提供 HTML 文件
  let filePath = HTML_FILE;

  if (req.url !== '/') {
    // 防止路径遍历
    const safePath = path.normalize(req.url).replace(/^\/+/, '');
    filePath = path.join(__dirname, safePath);
  }

  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
      res.end('404 Not Found');
      return;
    }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log('========================================');
  console.log('  热血乌龙球 - 文件控制服务器已启动');
  console.log('========================================');
  console.log(`  游戏地址:  http://localhost:${PORT}`);
  console.log(`  信号接口:  http://localhost:${PORT}/signal`);
  console.log(`  信号文件:  ${SIGNAL_FILE}`);
  console.log('');
  console.log('操作说明:');
  console.log('  - 向 movda.txt 写入 1 → 红方右移(Z键)');
  console.log('  - 向 movda.txt 写入 0 → 红方停止右移');
  console.log('  - 其他操作仍可用鼠标/键盘');
  console.log('');
  console.log('按 Ctrl+C 停止服务器');
  console.log('========================================');
});
