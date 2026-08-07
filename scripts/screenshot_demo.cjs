const { chromium } = require(
  "C:/Users/zhaor/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright"
);

(async () => {
  const browser = await chromium.launch({
    headless: true,
    executablePath: "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  });
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
  await page.goto(process.argv[2] || "http://127.0.0.1:8010", { waitUntil: "networkidle" });
  if (process.argv[6] === "example") {
    await page.locator("#upload-panel").evaluate((element) => { element.open = true; });
    await page.click(".example-image");
    await page.waitForFunction(() => document.querySelector("#upload-status")?.textContent.startsWith("已载入"));
  }
  if (process.argv[6] === "table") {
    await page.evaluate(() => {
      document.querySelector("#empty-state").classList.add("hidden");
      document.querySelector("#result-content").classList.remove("hidden");
      document.querySelector("#run-state").textContent = "COMPLETE";
      renderAnswer("## 不同时效预测\n\n| 时效 | 最大VIL | 强回波面积 | 结论 |\n|---|---:|---:|---|\n| +15分钟 | 255 | 2.8% | 强回波维持 |\n| +30分钟 | 247 | 2.3% | 范围缩小 |\n| +60分钟 | 231 | 1.7% | 逐渐减弱 |\n\n- 表格来自CasCast预测摘要。\n- 当前没有ground truth。");
      document.querySelector("#result-title").textContent = "未来强回波分析";
    });
  }
  if (process.argv[6] === "dataset") {
    await page.selectOption("#task", "dataset_evaluation");
    await page.locator("#mode").evaluate((element) => {
      element.value = "local";
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }
  if (process.argv[4] === "run") {
    if (process.argv[5] === "local") await page.locator("#mode").evaluate((element) => {
      element.value = "local";
      element.dispatchEvent(new Event("change", { bubbles: true }));
    });
    await page.click("#run-button");
    await page.waitForSelector("#result-content:not(.hidden)", { timeout: 30000 });
  }
  await page.screenshot({ path: process.argv[3] || "demo-screenshot.png", fullPage: true });
  await browser.close();
})();
