import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { createServer } from "node:http";
import { extname, join, normalize } from "node:path";
import { fileURLToPath } from "node:url";

const host = process.env.UI_HOST || "127.0.0.1";
const port = Number(process.env.UI_PORT || 3000);
const apiBase = process.env.API_BASE_URL || "http://127.0.0.1:8000";
const publicRoot = fileURLToPath(new URL("./public/", import.meta.url));

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
};

async function proxyApi(request, response) {
  const chunks = [];
  for await (const chunk of request) chunks.push(chunk);
  try {
    const upstream = await fetch(`${apiBase}${request.url}`, {
      method: request.method,
      headers: { "content-type": request.headers["content-type"] || "application/json" },
      body: ["GET", "HEAD"].includes(request.method) ? undefined : Buffer.concat(chunks),
    });
    response.writeHead(upstream.status, {
      "content-type": upstream.headers.get("content-type") || "application/json",
    });
    response.end(Buffer.from(await upstream.arrayBuffer()));
  } catch (error) {
    response.writeHead(502, { "content-type": "application/json" });
    response.end(JSON.stringify({ detail: `Python API unavailable: ${error.message}` }));
  }
}

async function serveStatic(request, response) {
  const urlPath = decodeURIComponent(new URL(request.url, `http://${host}`).pathname);
  const requested = urlPath === "/" ? "index.html" : urlPath.slice(1);
  const safePath = normalize(requested).replace(/^(\.\.(\/|\\|$))+/, "");
  const filePath = join(publicRoot, safePath);
  try {
    const info = await stat(filePath);
    if (!info.isFile()) throw new Error("not a file");
    response.writeHead(200, {
      "content-type": mimeTypes[extname(filePath)] || "application/octet-stream",
      "cache-control": "no-store",
    });
    createReadStream(filePath).pipe(response);
  } catch {
    response.writeHead(404, { "content-type": "text/plain; charset=utf-8" });
    response.end("Not found");
  }
}

createServer(async (request, response) => {
  if (request.url.startsWith("/api/")) return proxyApi(request, response);
  return serveStatic(request, response);
}).listen(port, host, () => {
  process.stdout.write(`FraudLens legacy integration UI: http://${host}:${port}\nPython API: ${apiBase}\n`);
});
