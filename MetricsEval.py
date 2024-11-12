import evaluate


class MetricsEval:
    """Class for evaluating using certain metrics."""

    def __init__(self, tokenizer, metric_type="wer"):
        """Initate the metric evaluation class.

        Args:
            tokenizer (Tokenizer): The tokenizer used for tokenizing the text
            metric_type (str, optional): The metric type to use. Defaults to "wer"
        """

        self.tokenizer = tokenizer
        self.metric_type = metric_type
        self.metric = evaluate.load(metric_type)

    def compute(self, pred):
        """Compute the metric.

        Args:
            pred (dict): The predictions
        """

        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Replace -100 with pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # Decode predictions and labels
        pred_str = self.tokenizer.batch_decode(
            pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(
            label_ids, skip_special_tokens=True)

        # Compute the metric
        metric_value = 100 * self.metric.compute(
            predictions=pred_str, references=label_str)

        return {self.metric_type: metric_value}
