import re


VALID_ICD_REGEX = re.compile(r"^[A-Z]\d{2,4}$")


def validate_icd_codes_list(icd_codes: list[str]) -> list[str]:
    valid_icd_codes = list()

    for icd in icd_codes:
        # Remove the optional dot between the first 3 characters and the rest of the string
        icd_without_dot = icd.replace('.', '').upper()

        matches = list(re.finditer(VALID_ICD_REGEX, icd_without_dot))
        assert(len(matches) <= 1)
        if len(matches) == 1:
            valid_icd_code = matches[0].group()
            valid_icd_codes.append(valid_icd_code)

    return valid_icd_codes
