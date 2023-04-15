# Transformer 库的使用

```
        output
          |
   ----------------
   |  Layer Norm  |
   ----------------
          |
 |--------+
 | ----------------
 | | Feed Forward |
 | ----------------
 |--------|
          |
   ----------------
   |  Layer Norm  |
   ----------------
          |
 |--------+
 |        |
 | ----------------
 | |  Attention   |
 | ----------------
 |  |     |     |
 |  @-Wk  @-Wv  @-Wq
 |  |     |     |
 |  -------------
 ---------|
          + --- | Positional Encoding |
          |
  -------------------
  | Input Embedding |
  -------------------
          |
          X
```

图：Transformer Encoder。`@`: Matrix Multiplication. `+`: Addition.

## `BertLMHeadModel`

`BertLMHeadModel` 是 Bert 模型的变体，将 Masked Language Model Loss 改为了 Language Model Loss。结构是 Transformer Encoder 的堆叠，在这里我们更关心在 `huggingface/transformers` 库中 `BertLMHeadModel` 的使用。

最基本的方式，是按照 `transformers` 入门中提到的，

```python
from transformers import BertTokenizer, BertModel
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

text = "The cat sat on the mat."
tokenized = tokenizer(text)
output = model(**tokenized)
```

这里的 `**tokenized` 是将 `tokenized` 中的所有 key-value pair 作为参数传入 `model`。这里的 `tokenized` 是一个字典，包含了 `input_ids`，`token_type_ids`，`attention_mask`。这也是按照 Bert 论文中描述的结构。这引出了 `BertModel` 的三个参数。

- `input_ids` (`torch.LongTensor` of shape `(batch_size, sequence_length)`): `token` 的下标。
- `token_type_ids` (`torch.LongTensor` of shape `(batch_size, sequence_length)`, optional) — Bert 中的 Contrastive Loss 用到的，前半句以及 `[SEP]` 对应的 `token` 都是 `0`，后半句对应的都是 `1`。
- `attention_mask` (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, optional) — 避免对 `token` 进行 `Attention` 或者 `Padding`。`1` 为被 `mask`，`0` 为未被 `mask`。==具体是什么作用？==

紧接着，我们有了直接输入 `embedding` 而不是 `token_id` 的需求。如 `BLIP-2` 中，

```python
query_output = self.Qformer.bert(
    query_embeds=query_tokens,
    encoder_hidden_states=image_embeds,
    encoder_attention_mask=image_atts,
    use_cache=True,
    return_dict=True,
)
```

- `inputs_embeds`: (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, optional) — 绕过 BERT 第一层的 `nn.Embedding`，直接将 `input_embeds` 输入 `Transformer Encoder`。
- `encoder_hidden_states`: (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, optional) — `Transformer Decoder` 形式下，作为 Cross-Attention 的 `key` 和 `value` 输入模型。
- `encoder_attention_mask`: (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, optional) — `Transformer Decoder` 形式下，作为 Cross-Attention 的 `mask` 输入模型。
- `use_cache`: (`bool`, optional) — 是否返回 `past_key_values`。`past_key_values` 是一个 `tuple`，包含了 `Transformer Encoder` 的每一层的 `key` 和 `value`。`past_key_values` 的长度为 `num_hidden_layers`，每个元素是一个 `tuple`，包含了 `key` 和 `value`，每个 `key` 和 `value` 的形状为 `(batch_size, num_heads, sequence_length, head_size)`。
- `return_dict`: (`bool`, optional) — 没啥好说的，`True` 则返回一个类似 `dict` 的 `dataclass`，`False` 则返回 `tuple`。

之后还有 `input_ids` 和 `query_embeds` 一起输入的情况。

```python
output_itm = self.Qformer.bert(
    text_ids_all,
    query_embeds=query_tokens_itm,
    attention_mask=attention_mask_all,
    encoder_hidden_states=image_embeds_all,
    encoder_attention_mask=image_atts_all,
    return_dict=True,
)
```

这种情况，源码里是这样处理的

```python
# BertModel.forward
if input_ids is not None:
    embeddings = self.word_embeddings(input_ids)
    if self.position_embedding_type == "absolute":
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = embeddings + position_embeddings

    if query_embeds is not None:
        embeddings = torch.cat((query_embeds, embeddings), dim=1)
```

就是将 `query_embeds` 和 `input_ids` 的 `embedding` 给 `concat` 起来。

另一个巧妙的用法是 `past_key_values`。官方文档中的解释是，

> to speed up decoding, you can feed the `past_key_values` parameter to the model. The `past_key_values` are made of the `encoder_outputs` of the model (see `encoder_outputs` below) you want to cache.

```python
class BertSelfAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if uni-directional self-attention (decoder) save Tuple(torch.Tensor, torch.Tensor) of
            # all previous decoder key/value_states. Further calls to uni-directional self-attention
            # can concat previous decoder key/value_states to current projected key/value_states (third "elif" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
```

这段代码中，第一个 `if-elif-else` 块，如果没有 `is_cross_attention` 但有 `past_key_value`，则走第三块，将 `past_key_value` 中的 `key` 和 `value` 与新生成的 `key` 和 `value` 进行拼接；如果都没有，走第四块，直接通过 Linear Layer 生成 `key` 和 `value`。再进入下一步的 `attention` 计算

其他属性，太多了，就不一一列举了。下附 `BLIP-2` 中三个 `forward` 函数的源码。

Stage 1

```python
def forward(self, samples):
    image = samples["image"]
    text = samples["text_input"]

    image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device
    )

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

    query_output = self.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        use_cache=True,
        return_dict=True,
    )

    image_feats = F.normalize(
        self.vision_proj(query_output.last_hidden_state), dim=-1
    )

    text_tokens = self.tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=self.max_txt_len,
        return_tensors="pt",
    ).to(image.device)
    text_output = self.Qformer.bert(
        text_tokens.input_ids,
        attention_mask=text_tokens.attention_mask,
        return_dict=True,
    )
    text_feat = F.normalize(
        self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
    )

    ###============== Image-text Contrastive ===================###
    image_feats_all = concat_all_gather(
        image_feats
    )  # [batch_size*num_gpu, num_query_tokens, embed_dim]
    text_feat_all = concat_all_gather(text_feat)  # [batch_size*num_gpu, embed_dim]

    sim_q2t = torch.matmul(
        image_feats.unsqueeze(1), text_feat_all.unsqueeze(-1)
    ).squeeze()
    # [batch_size, batch_size*num_gpu, num_query_tokens]

    # image-text similarity: aggregate across all query tokens
    sim_i2t, _ = sim_q2t.max(-1)
    sim_i2t = sim_i2t / self.temp

    # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
    sim_t2q = torch.matmul(
        text_feat.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
    ).squeeze()

    # text-image similarity: aggregate across all query tokens
    sim_t2i, _ = sim_t2q.max(-1)
    sim_t2i = sim_t2i / self.temp  # [batch_size, batch_size*num_gpu]

    rank = dist.get_rank()
    bs = image.size(0)
    targets = torch.linspace(rank * bs, rank * bs + bs - 1, bs, dtype=int).to(
        image.device
    )

    loss_itc = (
        F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
        + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
    ) / 2

    ###============== Image-text Matching ===================###
    text_input_ids_world = concat_all_gather(text_tokens.input_ids)
    text_attention_mask_world = concat_all_gather(text_tokens.attention_mask)
    image_embeds_world = all_gather_with_grad(image_embeds)
    with torch.no_grad():
        weights_t2i = F.softmax(sim_t2i, dim=1) + 1e-4
        weights_t2i[:, rank * bs : rank * bs + bs].fill_diagonal_(0)
        weights_i2t = F.softmax(sim_i2t, dim=1) + 1e-4
        weights_i2t[:, rank * bs : rank * bs + bs].fill_diagonal_(0)

    # select a negative image for each text
    image_embeds_neg = []
    for b in range(bs):
        neg_idx = torch.multinomial(weights_t2i[b], 1).item()
        image_embeds_neg.append(image_embeds_world[neg_idx])
    image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

    # select a negative text for each image
    text_ids_neg = []
    text_atts_neg = []
    for b in range(bs):
        neg_idx = torch.multinomial(weights_i2t[b], 1).item()
        text_ids_neg.append(text_input_ids_world[neg_idx])
        text_atts_neg.append(text_attention_mask_world[neg_idx])

    text_ids_neg = torch.stack(text_ids_neg, dim=0)
    text_atts_neg = torch.stack(text_atts_neg, dim=0)

    text_ids_all = torch.cat(
        [text_tokens.input_ids, text_tokens.input_ids, text_ids_neg], dim=0
    )  # pos, pos, neg
    text_atts_all = torch.cat(
        [text_tokens.attention_mask, text_tokens.attention_mask, text_atts_neg],
        dim=0,
    )

    query_tokens_itm = self.query_tokens.expand(text_ids_all.shape[0], -1, -1)
    query_atts_itm = torch.ones(query_tokens_itm.size()[:-1], dtype=torch.long).to(
        image.device
    )
    attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

    image_embeds_all = torch.cat(
        [image_embeds, image_embeds_neg, image_embeds], dim=0
    )  # pos, neg, pos
    image_atts_all = torch.ones(image_embeds_all.size()[:-1], dtype=torch.long).to(
        image.device
    )

    output_itm = self.Qformer.bert(
        text_ids_all,
        query_embeds=query_tokens_itm,
        attention_mask=attention_mask_all,
        encoder_hidden_states=image_embeds_all,
        encoder_attention_mask=image_atts_all,
        return_dict=True,
    )

    vl_embeddings = output_itm.last_hidden_state[:, : query_tokens_itm.size(1), :]
    vl_output = self.itm_head(vl_embeddings)
    logits = vl_output.mean(dim=1)

    itm_labels = torch.cat(
        [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
        dim=0,
    ).to(image.device)
    loss_itm = F.cross_entropy(logits, itm_labels)

    ##================= Image Captioning ========================##
    decoder_input_ids = text_tokens.input_ids.clone()
    decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
    labels = decoder_input_ids.masked_fill(
        decoder_input_ids == self.tokenizer.pad_token_id, -100
    )

    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
        image.device
    )
    attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
    lm_output = self.Qformer(
        decoder_input_ids,
        attention_mask=attention_mask,
        past_key_values=query_output.past_key_values,
        return_dict=True,
        labels=labels,
    )

    loss_lm = lm_output.loss

    return BlipOutput(
        loss=loss_itc + loss_itm + loss_lm,
        loss_itc=loss_itc,
        loss_itm=loss_itm,
        loss_lm=loss_lm,
    )
```

Stage 2 (Decoder-only)

```python
def forward(self, samples):
    image = samples["image"]
    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device
    )

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    inputs_opt = self.opt_proj(query_output.last_hidden_state)
    atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(image.device)

    self.opt_tokenizer.padding_side = "right"

    text = [t + "\n" for t in samples["text_input"]]

    opt_tokens = self.opt_tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=self.max_txt_len,
    ).to(image.device)

    targets = opt_tokens.input_ids.masked_fill(
        opt_tokens.input_ids == self.opt_tokenizer.pad_token_id, -100
    )
    if self.prompt:
        targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

    empty_targets = (
        torch.ones(atts_opt.size(), dtype=torch.long).to(image.device).fill_(-100)
    )
    targets = torch.cat([empty_targets, targets], dim=1)

    inputs_embeds = self.opt_model.model.decoder.embed_tokens(opt_tokens.input_ids)
    inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
    attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

    with self.maybe_autocast():
        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
    loss = outputs.loss

    return {"loss": loss}

```

Stage 2 (Encoder-decoder)

```python
def forward(self, samples):
    image = samples["image"]

    with self.maybe_autocast():
        image_embeds = self.ln_vision(self.visual_encoder(image))
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
        image.device
    )

    query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
    query_output = self.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    inputs_t5 = self.t5_proj(query_output.last_hidden_state)
    atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long).to(image.device)

    with self.maybe_autocast(dtype=torch.bfloat16):
        input_tokens = self.t5_tokenizer(
            samples["text_input"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        output_tokens = self.t5_tokenizer(
            samples["text_output"],
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)

        encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)

        targets = output_tokens.input_ids.masked_fill(
            output_tokens.input_ids == self.t5_tokenizer.pad_token_id, -100
        )

        inputs_embeds = self.t5_model.encoder.embed_tokens(input_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, inputs_embeds], dim=1)

        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_tokens.attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss

        return {"loss": loss}

```
