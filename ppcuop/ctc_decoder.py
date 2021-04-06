from typing import List


class CTCDecoder:
    def __init__(self, label_dict: dict, blank: int):
        """
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
        CTC Decoder
        :param text: Text to decode
        :return: result
        """
        return self.ctc_decoder(text, self.label_dict, self.blank)

    @staticmethod
    def ctc_decoder(text: List[int], label_dict: dict, blank: int):
        """
        :param text: Text to decode
        :param label_dict: A dictionary in the form of {0：“A”, 1: "B", ...}
        :param blank: Separator index
        :return: result
        """

        result = []
        cache_idx = -1
        for char in text:
            if char != blank and char != cache_idx:
                result.append(label_dict[char])
            cache_idx = char
        return "".join(result)


# Example
if __name__ == '__main__':
    text = [1, 0, 1, 1, 2, 1, 3, 3, 3, 0, 3, 1, 2]  # Expect: AABACCAB

    label_dict = {1: "A", 2: "B", 3: "C"}
    ctc_decoder = CTCDecoder(label_dict, blank=0)
    result = ctc_decoder.decoder(text)
    print(result)
