```typescript
require("dotenv").config();

import https from "https";
const token = process.env.WELM_APIKEY;

/**
 * model: string 必选，要使用的模型名称，当前支持的模型名称有medium、 large 和 xl
 * prompt: string 可选，默认值空字符串，给模型的提示
 * max_tokens: integer 可选，最多生成的token个数，默认值 16
 * temperature: number 可选 默认值 0.85，表示使用的sampling temperature，更高的temperature意味着模型具备更多的可能性。对于更有创造性的应用，可以尝试0.85以上，而对于有明确答案的应用，可以尝试0（argmax采样）。 建议改变这个值或top_p，但不要同时改变。
 * top_p: number 可选 默认值 0.95，来源于nucleus sampling，采用的是累计概率的方式。即从累计概率超过某一个阈值p的词汇中进行采样，所以0.1意味着只考虑由前10%累计概率组成的词汇。 建议改变这个值或temperature，但不要同时改变。
 * top_k: integer 可选 默认值50，从概率分布中依据概率最大选择k个单词，建议不要过小导致模型能选择的词汇少。
 * n: integer 可选 默认值 1 返回的序列的个数
 * echo: boolean 可选 默认值 false，是否返回prompt
 * stop: string 可选 默认值 null，停止符号。当模型当前生成的字符为stop中的任何一个字符时，会停止生成。若没有配置stop，当模型当前生成的token id 为end_id或生成的token个数达到max_tokens时，停止生成。合理配置stop可以加快推理速度、减少quota消耗。
 *
 * 最好的 prompt 设计原则是：第一，描述清楚；第二，例子具备代表性（多个例子更好）。
 */
interface WeLMRequest {
  model: "medium" | "large" | "xl";
  prompt?: string;
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  n?: number;
  echo?: boolean;
  stop?: string;
}

interface WeLMResponse {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    text: string;
    index: number;
    logprobs: number;
    finish_reason: string;
  }>;
}

const data: WeLMRequest = {
  prompt:
    "给自己的猫咪取个特色的名字。\n\n描述：美短 动漫角色\n名字：哆啦\n描述：英短 黑色 饮品\n名字：可乐\n描述：橘猫  黄色  食物\n名字：",
  model: "xl",
  max_tokens: 16,
  temperature: 0.85,
  top_p: 0.95,
  top_k: 10,
  n: 5,
  echo: false,
  stop: "\n",
};

const requestBuffer = Buffer.from(JSON.stringify(data));

console.log(requestBuffer.length);

const options: https.RequestOptions = {
  host: "welm.weixin.qq.com",
  port: 443,
  path: "/v1/completions",
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    "Content-Length": requestBuffer.length,
    Authorization: token,
  },
};

const req = https.request(options, function (res) {
  console.log(res.statusCode);
  res.on("data", function (result: Uint8Array) {
    const response: WeLMResponse = JSON.parse(result.toString());
    console.log(response);
  });
});

req.write(requestBuffer);

req.end();

req.on("error", function (e) {
  console.error(e);
});
```

预期返回

```json
{
  "id": "略",
  "object": "text_generation",
  "created": 0,
  "model": "xl",
  "choices": [
    { "text": "橘子", "index": 0, "logprobs": 0, "finish_reason": "finished" },
    { "text": "小橘", "index": 1, "logprobs": 0, "finish_reason": "finished" },
    { "text": "橙汁", "index": 2, "logprobs": 0, "finish_reason": "finished" },
    { "text": "薯条", "index": 3, "logprobs": 0, "finish_reason": "finished" },
    { "text": "薯片", "index": 4, "logprobs": 0, "finish_reason": "finished" }
  ]
}
```
