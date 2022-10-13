import re


VALID_CHOP_REGEX = re.compile(r"^\d[A-Z0-9]{1,5}$")


def validate_chop_codes_list(chop_codes: list[str]) -> list[str]:
    valid_chop_codes = list()

    for chop in chop_codes:
        # Remove the optional dots
        chop_without_dots = chop.replace('.', '').upper()

        matches = list(re.finditer(VALID_CHOP_REGEX, chop_without_dots))

        assert(len(matches) <= 1)
        if len(matches) == 1:
            valid_chop_code = matches[0].group()
            valid_chop_codes.append(valid_chop_code)
        else:
            invalid_chop_codes.append(chop)

    # invalid_chop_codes = [chop for chop in valid_chop_codes if chop not in valid_chop_codes]

    return valid_chop_codes, invalid_chop_codes
