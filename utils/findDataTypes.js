const findDataTypes = (value) => {
    if (/^-?\d+$/.test(value)) {
        return "integer";
    } else if (/^-?\d+\.\d+$/.test(value)) {
        return "float";
    } else if (/^(true|false)$/i.test(value)) {
        return "boolean";
    } else if (/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?/.test(value)) {
        return "date";
    } else {
        return "text";
    }
};

const convertStringToRespectiveTypes = (value) => {
    if (/^-?\d+$/.test(value)) {
        return Number(value);
    } else if (/^-?\d+\.\d+$/.test(value)) {
        return Number(value);
    } else if (/^(true|false)$/i.test(value)) {
        return Boolean(value);
    } else if (/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z?/.test(value)) {
        return Date(value);
    } else {
        return value;
    }
};

module.exports = {
    convertStringToRespectiveTypes,
    findDataTypes,
};
