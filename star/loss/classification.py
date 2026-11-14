"""Classification cross-entropy helpers for satellite images and drone sequences."""


def cal_loss1(outputs, labels, loss_func, batch_size, seq_len):
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            loss += loss_func(i, labels)
        loss = loss / len(outputs)
    else:
        loss = loss_func(outputs, labels)
    return loss


def cal_loss2(outputs, labels, loss_func, batch_size, seq_len):
    loss = 0
    if isinstance(outputs, list):
        for i in outputs:
            i = i.view(batch_size, seq_len, -1)  # 恢复序列维度
            for t in range(seq_len):
                loss += loss_func(i[:, t, :], labels)  # 每时间步的分类损失

        loss = loss / (len(outputs) * seq_len)  # 平均分类损失
    else:
        # 对单个输出也需要恢复序列维度并计算每个时间步的损失
        outputs = outputs.view(batch_size, seq_len, -1)
        for t in range(seq_len):
            loss += loss_func(outputs[:, t, :], labels)
        loss = loss / seq_len  # 平均时间步的损失
    return loss
