from typing import List


class CTCDecoder:
    def __init__(self, label_dict: dict, blank: int):
        """
        CTCDecoder
        :param label_dict: A dictionary in the form of {0：“A”, 1: "B", ...}
        :param blank: Separator index

        Example:
        (Expect [1, 0, 1, 1, 2, 1, 3, 3, 3, 0, 3, 1, 2] -> AABACCAB)

        text = [1, 0, 1, 1, 2, 1, 3, 3, 3, 0, 3, 1, 2]

        label_dict = {1: "A", 2: "B", 3: "C"}
        ctc_decoder = CTCDecoder(label_dict, blank=0)

        result = ctc_decoder.decoder(text) # -> 11213312
        """
        self.label_dict = label_dict
        self.blank = blank

    def decoder(self, text: List[int]):
        """
        decoder
        :param text: Text to decode
        :return: result
        """
        return self.ctc_decoder(text, self.blank, self.label_dict)

    def batch_decoder(self, texts: List[List[int]]):
        """
        batch_decoder
        :param texts: batch text to decode
        :return: result
        """
        return [self.ctc_decoder(text, self.blank, self.label_dict) for text in texts]

    def batch_decoder(self, texts: List[List[int]]):
        """
        batch_decoder
        :param texts: batch text to decode
        :return: result
        """
        return [self.ctc_decoder(text, self.label_dict, self.blank) for text in texts]

    @staticmethod
    def ctc_decoder(text: List[int], blank: int, label_dict: dict = None):
        """
        :param text: Text to decode
        :param blank: Separator index
        :param label_dict: A dictionary in the form of {0：“A”, 1: "B", ...}
        :return: result
        """

        result = []
        cache_idx = -1
        for char in text:
            if char != blank and char != cache_idx:
                if label_dict:
                    result.append(label_dict[char])
                else:
                    result.append(char)
            cache_idx = char
        return "".join(result) if label_dict else result


# Example
if __name__ == '__main__':
    _text = [1, 0, 1, 1, 2, 1, 3, 3, 3, 0, 3, 1, 2]  # Expect: AABACCAB

    _label_dict = {1: "A", 2: "B", 3: "C"}
    _ctc_decoder = CTCDecoder(_label_dict, blank=0)
    _result = _ctc_decoder.decoder(_text)
    print(_result)
