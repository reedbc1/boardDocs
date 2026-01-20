const fs = require("fs");
const cheerio = require("cheerio");


async function getBoardIds() {
    const response = await fetch("https://go.boarddocs.com/mo/slcl/Board.nsf/BD-GETMeetingsListForSEO?open&0.6159169630587711", {
    "headers": {
        "accept": "application/json, text/javascript, */*; q=0.01",
        "accept-language": "en-US,en;q=0.9",
        "sec-ch-ua": "\"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"144\", \"Google Chrome\";v=\"144\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "cookie": "SessionID=A475F12D893932AE307737759F211C77DEC93127",
        "Referer": "https://go.boarddocs.com/mo/slcl/Board.nsf/Public"
    },
    "body": null,
    "method": "GET"
    });

    const data = await response.json();
    const jsonData = JSON.stringify(data);
    await writeFile('meetingIds.json', jsonData);
    return data;
}


async function writeFile(fileName, jsonData) {
    return new Promise((resolve, reject) => {
        fs.writeFile(fileName, jsonData, function(err) {
            if (err) {
                reject(err);
            } else {
                resolve();
            }
        });
    });
}


function readFile(filename) {
    const fs = require('node:fs');
    try {
    const data = fs.readFileSync(filename, 'utf8');
    return data;
    } catch (err) {
    console.error(err);
    }

}


async function downloadAgenda(id) {
    // fetch request for agenda with id
    const response = await fetch("https://go.boarddocs.com/mo/slcl/Board.nsf/Download-AgendaDetailed?open&%22%20+%20Math.random()", {
    "headers": {
        "accept": "*/*",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "sec-ch-ua": "\"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"144\", \"Google Chrome\";v=\"144\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "cookie": "SessionID=A475F12D893932AE307737759F211C77DEC93127",
        "Referer": "https://go.boarddocs.com/mo/slcl/Board.nsf/Public"
    },
    "body": `id=${id}&current_committee_id=AAUJEW4CE322`,
    "method": "POST"
    });

    try {
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`);
        }
        const html = await response.text();
        return html;
    } catch (error) {
        console.error(`Failed to download agenda for ${id}: ${error.message}`);
        return null;
    }
}


async function downloadMinutes(id) {
    const response = await fetch("https://go.boarddocs.com/mo/slcl/Board.nsf/BD-GetMinutes?open&login0.4331538422449709", {
    "headers": {
        "accept": "text/html, */*; q=0.01",
        "accept-language": "en-US,en;q=0.9",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "sec-ch-ua": "\"Not(A:Brand\";v=\"8\", \"Chromium\";v=\"144\", \"Google Chrome\";v=\"144\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Windows\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "x-requested-with": "XMLHttpRequest",
        "cookie": "SessionID=A475F12D893932AE307737759F211C77DEC93127",
        "Referer": "https://go.boarddocs.com/mo/slcl/Board.nsf/Public"
    },
    "body": `id=${id}&current_committee_id=AAUJEW4CE322`,
    "method": "POST"
    });

    try {
        if (!response.ok) {
            throw new Error(`Response status: ${response.status}`);
        }
        const html = await response.text();
        return html;
    } catch (error) {
        console.error(`Failed to download minutes for ${id}: ${error.message}`);
        return null;
    }
}


async function main() {
    // Load processed IDs
    let processedIds = [];
    try {
        const data = readFile('processedIds.json');
        processedIds = JSON.parse(data);
    } catch (e) {
        // File doesn't exist, start with empty array
    }

    // Fetch meeting IDs
    const meetingData = await getBoardIds();

    // Extract all IDs
    const allIds = meetingData.map(arr => {
        console.log(typeof arr.Date);
        return {
            'name': arr.Name, 
            'description': arr.Description,
            'unique': arr.Unique,
            'date': arr.Date
        }
    });

    // Filter to only new IDs
    const newIds = allIds.filter(dict => !processedIds.includes(dict.date));

    // Process new IDs
    for (let id of newIds) {
        console.log(`Processing id: ${id.unique}`);

        // Download agenda
        const agendaHtml = await downloadAgenda(id.unique);
        if (agendaHtml) {
            const text = extractText(agendaHtml);

            console.log(typeof id.date);
            date = new Date(id.date);

            if (text.trim()) {
                const sanitizedDate = id.date.split('T')[0]; // Gets '2025-12-18'
                const metadata = JSON.stringify({
                    name: id.name,
                    description: id.description,
                    unique: id.unique,
                    date: id.date
                }, null, 2);
                const contentWithMetadata = `${metadata}\n\n---\n\n${text}`;
                const textPath = `agendas\\${sanitizedDate}.txt`;
                await writeFile(textPath, contentWithMetadata);
                console.log(`Saved agenda text for ${id.date}`);
            } else {
                console.log(`No text in agenda for ${id.date}`);
            }
        }

        // Download minutes
        const minutesHtml = await downloadMinutes(id.unique);
        if (minutesHtml) {
            const text = extractText(minutesHtml);
            if (text.trim()) {
                const sanitizedDate = id.date.split('T')[0]; // Gets '2025-12-18'
                const metadata = JSON.stringify({
                    name: id.name,
                    description: id.description,
                    unique: id.unique,
                    date: id.date
                }, null, 2);
                const contentWithMetadata = `${metadata}\n\n---\n\n${text}`;
                const textPath = `minutes\\${sanitizedDate}.txt`;
                await writeFile(textPath, contentWithMetadata);
                console.log(`Saved minutes text for ${id.date}`);
            } else {
                console.log(`No text in minutes for ${id.date}`);
            }
        }
    }

    // Update processed IDs
    processedIds.push(...newIds);
    await writeFile('processedIds.json', JSON.stringify(processedIds));
}

main().catch(console.error);

function extractText(html) {
    
    // load into cheerio
    const $ = cheerio.load(html);

    // get text only
    let text = $("body").text();

    // clean it up
    text = text
    .replace(/\u00a0/g, " ")        // non-breaking spaces
    .replace(/\s+\n/g, "\n")        // trailing spaces
    .replace(/\n{3,}/g, "\n\n")     // collapse blank lines
    .trim();

    return text;
}

